from spin_trainer import *
from trainer_callback import *
from spin_data_collator import *
from spin_dataset import *
from utils import *
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
from dataclasses import asdict
import logging
import gc
import shutil
import torch
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup(cfg):
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.synthetic_cache_dir)
    ensure_dir(cfg.checkpoints_dir)
    ensure_dir(cfg.tensorboard_dir)
    if cfg.enable_profiler:
        ensure_dir(cfg.profile_dir)
    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
    set_seed(cfg.seed)
    logger.info(
        f"setup() complete — seed={cfg.seed}, output_dir={cfg.output_dir}")


def load_tokenizer_and_data(cfg):
    tokenizer = load_tokenizer(cfg)
    base_ds = load_base_dataset_fixed(
        dataset_name=cfg.dataset_name if cfg.dataset_name else None,
        dataset_config_name=cfg.dataset_config_name if cfg.dataset_config_name else None,
        split=cfg.train_split,
        data_path=cfg.data_path if cfg.data_path else None,
        limit=cfg.max_data_load if cfg.max_data_load > 0 else None,
        prompt_field=cfg.prompt_field,
        response_field=cfg.response_field,
    )
    # Materialise in fixed index order — no shuffling.
    base_rows = [base_ds[i] for i in range(len(base_ds))]
    logger.info(
        f"Dataset loaded: {len(base_rows)} rows in deterministic order.")
    return tokenizer, base_rows


# ── Checkpoint cleanup ────────────────────────────────────────────────────────

def _cleanup_trainer_checkpoints(directory: str):
    """Delete HF Trainer checkpoint-N subdirs from a directory."""
    if not os.path.isdir(directory):
        return
    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            shutil.rmtree(entry.path)
            logger.info(f"Removed HF checkpoint: {entry.path}")


# ── Batch resume detection ────────────────────────────────────────────────────

def _find_start_batch(cfg, iteration, total_batches):
    """
    Return the index of the first batch where not all 3 steps are complete.
    Complete = synth file valid AND logprobs file valid AND .done sentinel exists.
    Since batches are processed sequentially, the first incomplete batch is always
    the resumption point.
    """
    for k in range(total_batches):
        synth_ok = file_valid(synth_path(cfg, iteration, k))
        logp_ok = file_valid(logprobs_path(cfg, iteration, k))
        train_ok = os.path.exists(batch_done_path(cfg, iteration, k))
        if not (synth_ok and logp_ok and train_ok):
            logger.info(
                f"  First incomplete batch for iter_{iteration}: k={k} "
                f"(synth={synth_ok}, logprobs={logp_ok}, train_done={train_ok})")
            return k
    logger.info(
        f"  All {total_batches} batches already complete for iter_{iteration}.")
    return total_batches


def _init_train_model(prev_model, cfg, iteration, start_batch):
    """
    Return the initial trainable model for this iteration.
    - start_batch == 0: wrap prev_model with LoRA.
    - start_batch  > 0: load the merged model saved after batch start_batch-1,
                        then wrap with fresh LoRA adapters.
    """
    if start_batch == 0:
        logger.info(
            "  init_train_model: batch 0 — converting prev_model to trainable.")
        return make_trainable(prev_model, cfg)

    last_dir = batch_train_dir(cfg, iteration, start_batch - 1)
    logger.info(
        f"  init_train_model: resuming at batch {start_batch} — "
        f"loading base model from {last_dir}.")
    base = load_causal_lm(last_dir, cfg, trainable=False).to(cfg.device)
    return make_trainable(base, cfg)


# ── Per-batch step 1: synthetic generation ────────────────────────────────────

def _step_synth(prev_model, tokenizer, chunk, cfg, iteration, k):
    """
    Generate synthetic responses for `chunk`.  Skips if the output file already
    exists.  prev_model is moved to GPU for generation then back to CPU so the
    GPU is free for the subsequent training step.
    """
    path = synth_path(cfg, iteration, k)
    if file_valid(path):
        rows = load_jsonl(path)
        logger.info(
            f"    [1/3 SKIP] synth: {len(rows)} rows loaded from {path}")
        return rows

    logger.info(f"    [1/3 RUN ] synth: generating for {len(chunk)} rows...")
    prev_model.to(cfg.device)
    rows = generate_synthetic_responses(prev_model, tokenizer, chunk, cfg)
    prev_model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_save_jsonl(path, rows)
    logger.info(f"    [1/3 DONE] synth: {len(rows)} rows saved → {path}")
    return rows


# ── Per-batch step 2: ref logprob scoring ─────────────────────────────────────

def _step_logprobs(prev_model, tokenizer, synth_rows, cfg, iteration, k):
    """
    Score (chosen, rejected) pairs under the frozen prev_model.  Skips if the
    output file already exists.  GPU management mirrors _step_synth.
    """
    path = logprobs_path(cfg, iteration, k)
    if file_valid(path):
        lps = load_jsonl(path)
        logger.info(
            f"    [2/3 SKIP] logprobs: {len(lps)} rows loaded from {path}")
        return lps

    logger.info(f"    [2/3 RUN ] logprobs: scoring {len(synth_rows)} rows...")
    prev_model.to(cfg.device)
    lps = compute_ref_logprobs(prev_model, tokenizer, synth_rows, cfg)
    prev_model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_save_jsonl(path, lps)
    logger.info(f"    [2/3 DONE] logprobs: {len(lps)} rows saved → {path}")
    return lps


# ── Per-batch step 3: training ────────────────────────────────────────────────

def _step_train(train_model, synth_rows, ref_lps, tokenizer, cfg,
                iteration, k, total_batches, summary_callback):
    """
    Train train_model on this batch's data.

    If .done already exists (batch was completed in a previous run), the merged
    model is loaded from disk and returned — the caller needs it to initialise
    the next batch's trainable model.

    Otherwise:
      - Creates SPINDataset + SPINTrainer for just this batch.
      - Resumes from an HF mid-batch checkpoint if one exists (handles kills
        that happen during the training loop itself).
      - Merges LoRA adapters, saves the full model to batch_dir.
      - Writes .done sentinel.

    Returns the merged (non-LoRA) model so the caller can call make_trainable()
    for the next batch without reloading from disk.
    """
    batch_dir = batch_train_dir(cfg, iteration, k)
    done_path = batch_done_path(cfg, iteration, k)
    ensure_dir(batch_dir)

    if os.path.exists(done_path):
        logger.info(f"    [3/3 SKIP] train: .done found at {done_path}")
        # Load merged model so the caller can continue with the next batch.
        model = load_causal_lm(batch_dir, cfg, trainable=False).to(cfg.device)
        return model

    logger.info(
        f"    [3/3 RUN ] train: {len(synth_rows)} rows, "
        f"batch {k + 1}/{total_batches}, iter {iteration}.")

    dataset = SPINDataset(synth_rows, tokenizer, cfg, ref_logprobs=ref_lps)
    collator = SPINDataCollator(tokenizer)

    spin_lambda = get_iteration_lambda(cfg, iteration)
    lr = get_iteration_lr(cfg, iteration)
    tb_log_dir = os.path.join(
        cfg.tensorboard_dir, f"iter_{iteration}", f"batch_{k:06d}")

    args = build_training_args(cfg, batch_dir, lr, logging_dir=tb_log_dir)

    callbacks = [
        TensorBoardCallbackExtended(
            log_dir=tb_log_dir,
            log_histograms=False,
            cfg=cfg,
            tokenizer=tokenizer,
            iteration=iteration,
        ),
        MemoryProbeCallback(writer=summary_callback.writer),
        TensorBoardParameterStatsCallback(
            cfg=cfg,
            run_name=f"iter_{iteration}_batch_{k:06d}",
        ),
        summary_callback,
    ]
    if cfg.enable_profiler:
        callbacks.append(TorchProfilerCallback(cfg=cfg, spin_iteration=iteration, tb_writer=summary_callback.writer))

    trainer_cls = RMSPropSPINTrainer if cfg.optimizer.lower() == "rmsprop" else SPINTrainer
    trainer = trainer_cls(
        model=train_model,
        spin_lambda=spin_lambda,
        loss_type=cfg.loss_type,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Resume from an HF Trainer checkpoint if the training step was interrupted
    # mid-epoch (e.g. laptop shut down during backward pass).
    resume_ckpt = get_last_checkpoint(batch_dir)
    if resume_ckpt:
        logger.info(f"    Mid-batch HF checkpoint detected: {resume_ckpt}")

    log_memory(f"iter{iteration}_batch{k}_before_train")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    log_memory(f"iter{iteration}_batch{k}_after_train")

    # Merge LoRA into base weights and save the full model.
    train_model = merge_lora_and_get_base(train_model, cfg)
    trainer.model = train_model
    trainer.save_model(batch_dir)

    # Remove the HF Trainer checkpoint-N subdirs (model is already saved above).
    _cleanup_trainer_checkpoints(batch_dir)

    open(done_path, "w").close()
    logger.info(
        f"    [3/3 DONE] train: model saved → {batch_dir}, .done written.")
    return train_model


# ── Iteration orchestration ───────────────────────────────────────────────────

def run_iteration(prev_model, tokenizer, base_rows, cfg, iteration, iter_dir, summary_callback):
    """
    Run one SPIN iteration by processing the full dataset in data_batch_size chunks.

    Within each chunk (batch) the three steps run in order:
      1. Synthetic generation  — prev_model generates rejected responses.
      2. Ref logprob scoring   — prev_model scores chosen + rejected.
      3. SPIN training         — train_model is updated on this batch's data.

    Every step saves its output atomically before the next step begins, so a
    thermal shutdown can be recovered at the exact step boundary.

    At the end of the iteration the final merged model is copied to iter_dir
    and an iteration-level .done sentinel is written.
    """
    logger.info(f"=== run_iteration() — SPIN iteration {iteration} ===")

    B = cfg.data_batch_size
    total = len(base_rows)
    total_batches = (total + B - 1) // B
    logger.info(
        f"  {total} rows, data_batch_size={B}, total_batches={total_batches}.")

    start_batch = _find_start_batch(cfg, iteration, total_batches)
    if start_batch == total_batches:
        logger.info(
            f"  All batches already done for iter_{iteration} — nothing to process.")
    else:
        summary_callback.set_iteration(
            iteration,
            spin_lambda=get_iteration_lambda(cfg, iteration),
            dataset_size=len(base_rows),
        )
        train_model = _init_train_model(
            prev_model, cfg, iteration, start_batch)

        for k in range(start_batch, total_batches):
            chunk = base_rows[k * B: (k + 1) * B]
            lo, hi = k * B, k * B + len(chunk) - 1
            logger.info(
                f"  ┌─ Batch {k + 1}/{total_batches}  "
                f"(dataset rows {lo}–{hi}, {len(chunk)} rows) ─────────────")

            synth_rows = _step_synth(
                prev_model, tokenizer, chunk, cfg, iteration, k)
            ref_lps = _step_logprobs(
                prev_model, tokenizer, synth_rows, cfg, iteration, k)
            train_model = _step_train(
                train_model, synth_rows, ref_lps, tokenizer, cfg,
                iteration, k, total_batches, summary_callback)

            logger.info(f"  └─ Batch {k + 1}/{total_batches} complete.")

            # After _step_train the model is the merged base (no LoRA adapters).
            # Re-wrap with fresh LoRA for the next batch, unless this was the last.
            if k < total_batches - 1:
                train_model = make_trainable(train_model, cfg)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Save final iteration model ─────────────────────────────────────────
        # The model from the last batch is already the merged base model.
        # Copy it to iter_dir so the next iteration's prev_model loads from there.
        last_batch_dir = batch_train_dir(cfg, iteration, total_batches - 1)
        if last_batch_dir != iter_dir:
            logger.info(f"  Saving final iteration model to {iter_dir}...")
            train_model.save_pretrained(iter_dir)

        free_model(train_model)

    tokenizer.save_pretrained(iter_dir)
    done_path = os.path.join(iter_dir, ".done")
    open(done_path, "w").close()
    logger.info(
        f"  Iteration {iteration} .done sentinel written → {done_path}")
    logger.info(f"run_iteration() complete — iteration {iteration}.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗")
    logger.info(
        "║              SPIN Training — main() starting                 ║")
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝")

    cfg = parse_args()
    logger.info(
        f"  model={cfg.model_name_or_path}  dataset={cfg.data_path or cfg.dataset_name}"
        f"  iterations={cfg.num_iterations}  data_batch_size={cfg.data_batch_size}"
        f"  optimizer={cfg.optimizer}  lr={cfg.learning_rate:.2e}"
        f"  use_lora={cfg.use_lora}(r={cfg.lora_r})")

    setup(cfg)
    tokenizer, base_rows = load_tokenizer_and_data(cfg)

    start_iteration = find_start_iteration(cfg)
    logger.info(
        f"Starting at iteration {start_iteration} "
        f"({'fresh run' if start_iteration == 0 else 'resuming'}).")

    global_tb_dir = os.path.join(cfg.tensorboard_dir, "global")
    ensure_dir(global_tb_dir)

    summary_callback = SPINIterationSummaryCallback(log_dir=global_tb_dir, cfg=cfg)
    logger.info(f"Global TensorBoard writer at: {global_tb_dir}")

    for iteration in range(start_iteration, cfg.num_iterations):
        remaining = cfg.num_iterations - iteration
        logger.info("")
        logger.info(
            f"╔══ SPIN ITERATION {iteration}/{cfg.num_iterations - 1} "
            f"({remaining} remaining) ══════════════════════════════╗")

        iter_dir = os.path.join(cfg.checkpoints_dir, f"iter_{iteration}")
        ensure_dir(iter_dir)

        prev_model_path = (
            cfg.model_name_or_path if iteration == 0
            else os.path.join(cfg.checkpoints_dir, f"iter_{iteration - 1}")
        )
        logger.info(f"  Loading π_prev from: {prev_model_path}")
        log_memory(f"before_load_iter{iteration}")
        prev_model = load_causal_lm(
            prev_model_path, cfg, trainable=False).to("cpu")
        log_memory(f"after_load_iter{iteration}")

        run_iteration(prev_model, tokenizer, base_rows, cfg,
                      iteration, iter_dir, summary_callback)

        free_model(prev_model)
        logger.info(
            f"╚══ SPIN ITERATION {iteration} COMPLETE ══════════════════════════════════╝")

    summary_callback.close()
    logger.info("")
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗")
    logger.info(
        "║              SPIN Training — main() complete                 ║")
    logger.info(
        f"║  Final: {os.path.join(cfg.checkpoints_dir, f'iter_{cfg.num_iterations - 1}')}")
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
