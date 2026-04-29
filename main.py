from spin_trainer import *
from trainer_callback import *
from spin_data_collator import *
from spin_dataset import *
from utils import *
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
from dataclasses import asdict
import logging
import random
import gc
import torch
import os

# Must be set before torch is imported so the CUDA allocator picks it up.
# expandable_segments: reduces fragmentation when many tensors of varying sizes
# are allocated/freed rapidly (typical during forward+backward of variable-length sequences).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


# ── Setup ────────────────────────────────────────────────────────────────────

def setup(cfg):
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.synthetic_cache_dir)
    ensure_dir(cfg.checkpoints_dir)
    if cfg.enable_profiler:
        ensure_dir(cfg.profile_dir)
    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
    set_seed(cfg.seed)
    logger.info("Config parsed, seed set, output directories ready.")


def load_tokenizer_and_data(cfg):
    tokenizer = load_tokenizer(cfg)
    base_ds = load_base_dataset_fixed(
        dataset_name=cfg.dataset_name if cfg.dataset_name else None,
        dataset_config_name=cfg.dataset_config_name if cfg.dataset_config_name else None,
        split=cfg.train_split,
        data_path=cfg.data_path if cfg.data_path else None,
        limit=cfg.max_data_load if cfg.max_data_load > 0 else None,
    )
    base_rows = [base_ds[i] for i in range(len(base_ds))]
    logger.info(
        f"Loaded full dataset: {len(base_rows)} rows (sampled per-iteration).")
    return tokenizer, base_rows


def restore_accumulated_rows(cfg, start_iteration):
    """Reload synthetic rows from previous iterations so the curriculum is intact on resume."""
    accumulated_rows = []
    if start_iteration > 0 and cfg.accumulate_previous_synthetic and cfg.save_synthetic_jsonl:
        for prev_iter in range(start_iteration):
            prev_synth_path = os.path.join(
                cfg.synthetic_cache_dir, f"iter_{prev_iter}.jsonl")
            if os.path.exists(prev_synth_path):
                prev_rows = load_jsonl(prev_synth_path)
                accumulated_rows.extend(prev_rows)
                logger.info(f"  Restored {len(prev_rows)} rows from iter_{prev_iter}.jsonl "
                            f"(accumulated total: {len(accumulated_rows)}).")
    return accumulated_rows


# ── Per-iteration data helpers ────────────────────────────────────────────────

def get_or_generate_synthetic(prev_model, tokenizer, base_rows, cfg, iteration):
    """Return (synthetic_rows, synth_cache_hit). Generates and caches atomically on a miss."""
    limit = cfg.synthetic_examples_per_iteration if cfg.synthetic_examples_per_iteration > 0 else len(
        base_rows)
    synth_path = os.path.join(cfg.synthetic_cache_dir,
                              f"iter_{iteration}.jsonl")
    synth_cache_hit = cfg.save_synthetic_jsonl and os.path.exists(synth_path)

    if synth_cache_hit:
        # Reuse synthetic data from a previous (possibly interrupted) run of this
        # iteration. This is critical for correctness: if training was killed
        # mid-way and left a checkpoint, resuming requires the exact same dataset
        # that was used when the checkpoint was created. Regenerating with a freshly
        # shuffled base_rows would produce different rows and corrupt the resume.
        synthetic_rows = load_jsonl(synth_path)
        logger.info(f"Loaded {len(synthetic_rows)} cached synthetic rows from {synth_path} "
                    f"(skipping generation).")
    else:
        # Use a seed derived from (global seed + iteration) so the sample is
        # different each iteration but fully reproducible on resume.
        iter_rng = random.Random(cfg.seed + iteration)
        rows_for_generation = iter_rng.sample(
            base_rows, min(limit, len(base_rows)))
        logger.info(f"Sampled {len(rows_for_generation)} rows for synthetic generation "
                    f"(seed: {cfg.seed + iteration}).")
        logger.info("Generating synthetic responses via prev_model...")
        synthetic_rows = generate_synthetic_responses(
            prev_model, tokenizer, rows_for_generation, cfg)
        logger.info(
            f"Synthetic generation complete: {len(synthetic_rows)} responses produced.")
        if cfg.save_synthetic_jsonl:
            atomic_save_jsonl(synth_path, synthetic_rows)

    return synthetic_rows, synth_cache_hit


def get_or_compute_ref_logprobs(prev_model, tokenizer, train_rows, cfg, iteration, synth_cache_hit):
    """Return ref_logprobs, loading from disk when the synth cache was hit and the file is valid."""
    ref_logprobs_path = os.path.join(
        cfg.synthetic_cache_dir, f"iter_{iteration}_ref_logprobs.jsonl")

    if synth_cache_hit and os.path.exists(ref_logprobs_path):
        ref_logprobs = load_jsonl(ref_logprobs_path)
        if len(ref_logprobs) != len(train_rows):
            # Stale or partially-written cache (e.g. killed during a previous save).
            logger.warning(
                f"Cached ref log-probs count ({len(ref_logprobs)}) != train rows "
                f"({len(train_rows)}); recomputing."
            )
            ref_logprobs = compute_ref_logprobs(
                prev_model, tokenizer, train_rows, cfg)
            if cfg.save_synthetic_jsonl:
                atomic_save_jsonl(ref_logprobs_path, ref_logprobs)
                logger.info(f"Ref log-probs re-cached to {ref_logprobs_path}.")
        else:
            logger.info(f"Loaded {len(ref_logprobs)} cached ref log-probs from {ref_logprobs_path} "
                        f"(skipping computation).")
    else:
        ref_logprobs = compute_ref_logprobs(
            prev_model, tokenizer, train_rows, cfg)
        if cfg.save_synthetic_jsonl:
            atomic_save_jsonl(ref_logprobs_path, ref_logprobs)
            logger.info(f"Ref log-probs cached to {ref_logprobs_path}.")

    return ref_logprobs


# ── Training ──────────────────────────────────────────────────────────────────

def build_callbacks(cfg, iter_dir, iteration, summary_cb, tokenizer):
    callbacks = (
        [TorchProfilerCallback(
            cfg, spin_iteration=iteration, tb_writer=summary_cb.writer)]
        if cfg.enable_profiler else []
    )
    callbacks.append(
        TensorBoardCallbackExtended(
            log_dir=os.path.join(iter_dir, "tb_logs"),
            log_histograms=False,
            cfg=cfg,
            tokenizer=tokenizer,
            iteration=iteration,
        ),
    )
    if cfg.log_trainable_parameters:
        callbacks.append(
            TensorBoardParameterStatsCallback(
                cfg=cfg,
                run_name=f"spin_iter_{iteration}",
            )
        )
    # Pass the global writer so system/ memory metrics also appear in tb_global/
    callbacks.append(MemoryProbeCallback(writer=summary_cb.writer))
    # summary_cb must be last: its on_train_end writes the cross-iteration summary
    # after all other callbacks have finished their on_train_end
    callbacks.append(summary_cb)
    return callbacks


def run_training(train_model, train_rows, ref_logprobs, tokenizer, cfg, iter_dir, iteration, summary_cb):
    """Build dataset, initialize trainer, train, merge LoRA, save checkpoint. Returns saved model."""
    # Pre-tokenizes all rows upfront; raw text no longer needed after this point.
    dataset = SPINDataset(train_rows, tokenizer, cfg,
                          ref_logprobs=ref_logprobs)
    n_train_rows = len(train_rows)
    del train_rows, ref_logprobs
    gc.collect()

    collator = SPINDataCollator(tokenizer)
    spin_lambda = get_iteration_lambda(cfg, iteration)
    lr = get_iteration_lr(cfg, iteration)
    args = build_training_args(cfg, iter_dir, lr)
    logger.info(f"Dataset pre-tokenized ({len(dataset)} examples). "
                f"Training args built (lr={lr}, lambda={spin_lambda}).")

    trainer_cls = RMSPropSPINTrainer if cfg.optimizer.lower() == "rmsprop" else SPINTrainer
    summary_cb.set_iteration(
        iteration, spin_lambda=spin_lambda, dataset_size=n_train_rows)
    callbacks = build_callbacks(
        cfg, iter_dir, iteration, summary_cb, tokenizer)
    print(len(dataset), dataset)
    logger.info("Callbacks registered. Initializing trainer.")

    log_memory("before_trainer_init")
    trainer = trainer_cls(
        model=train_model,
        spin_lambda=spin_lambda,
        loss_type=cfg.loss_type,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        # tokenizer=tokenizer,
        callbacks=callbacks,
    )
    log_memory("after_trainer_init")

    # Always resume from the latest HuggingFace checkpoint inside iter_dir if one exists.
    # On the first run of an iteration iter_dir has no checkpoint subdirs, so this is None.
    # On a restart after a kill mid-iteration, the trainer picks up where it left off.
    resume_ckpt = get_last_checkpoint(iter_dir)
    if resume_ckpt:
        logger.info(f"Resuming mid-iteration from checkpoint: {resume_ckpt}")
    logger.info("Starting training...")

    log_memory("before_trainer_train")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info(
        "Training complete. Merging LoRA adapters and saving model checkpoint.")

    # If LoRA was used, merge adapters into the base weights before saving.
    # This produces a plain AutoModelForCausalLM checkpoint with no PEFT
    # dependency so the next iteration's load_causal_lm() just works.
    train_model = merge_lora_and_get_base(train_model, cfg)
    # ensure trainer saves the unwrapped base model, not the PEFT wrapper
    trainer.model = train_model
    trainer.save_model(iter_dir)
    logger.info(f"Model checkpoint saved to {iter_dir}.")
    return train_model


# ── Iteration orchestration ───────────────────────────────────────────────────

def run_iteration(prev_model, tokenizer, base_rows, accumulated_rows, cfg, iteration, iter_dir, summary_cb):
    """Run one complete SPIN iteration. Returns updated accumulated_rows."""
    synthetic_rows, synth_cache_hit = get_or_generate_synthetic(
        prev_model, tokenizer, base_rows, cfg, iteration
    )

    if cfg.accumulate_previous_synthetic:
        accumulated_rows = synthetic_rows if iteration == 0 else accumulated_rows + synthetic_rows
        train_rows = accumulated_rows
    else:
        train_rows = synthetic_rows
    del synthetic_rows
    logger.info(f"Training data assembled: {len(train_rows)} rows total "
                f"(accumulate={cfg.accumulate_previous_synthetic}).")

    # Score under frozen prev_model before converting it to the trainable π_θ.
    ref_logprobs = get_or_compute_ref_logprobs(
        prev_model, tokenizer, train_rows, cfg, iteration, synth_cache_hit
    )
    train_model = make_trainable(prev_model, cfg)
    del prev_model

    train_model = run_training(
        train_model, train_rows, ref_logprobs, tokenizer, cfg, iter_dir, iteration, summary_cb
    )

    tokenizer.save_pretrained(iter_dir)
    logger.info("Tokenizer saved. Writing .done sentinel.")

    # Write a sentinel so find_start_iteration() can skip this iteration on restart.
    # Written AFTER both model and tokenizer are fully saved.
    open(os.path.join(iter_dir, ".done"), "w").close()
    logger.info(
        f"Iteration {iteration} complete. Sentinel written to {iter_dir}/.done")

    free_model(train_model)
    logger.info("GPU memory freed. Ready for next iteration.")

    return accumulated_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()
    setup(cfg)

    tokenizer, base_rows = load_tokenizer_and_data(cfg)

    start_iteration = find_start_iteration(cfg)
    if start_iteration > 0:
        logger.info(f"Resuming: iterations 0–{start_iteration - 1} already complete. "
                    f"Starting at iteration {start_iteration}.")

    accumulated_rows = restore_accumulated_rows(cfg, start_iteration)

    # Created once outside the loop so the x-axis spans all SPIN iterations.
    # Writes to tb_global/ so it shows up as a separate run in TensorBoard,
    # separate from the per-iteration tb_logs/ runs.
    global_tb_dir = os.path.join(cfg.output_dir, "tb_global")
    ensure_dir(global_tb_dir)
    summary_cb = SPINIterationSummaryCallback(log_dir=global_tb_dir, cfg=cfg)
    logger.info("Global TensorBoard writer initialized. Starting SPIN loop.")

    for iteration in range(start_iteration, cfg.num_iterations):
        logger.info(f"========== SPIN ITERATION {iteration} ==========")
        iter_dir = os.path.join(cfg.checkpoints_dir, f"iter_{iteration}")
        ensure_dir(iter_dir)

        prev_model_path = (
            cfg.model_name_or_path if iteration == 0
            else os.path.join(cfg.checkpoints_dir, f"iter_{iteration - 1}")
        )
        logger.info(f"prev_model_path: {prev_model_path}")
        prev_model = load_causal_lm(
            prev_model_path, cfg, trainable=False).to(cfg.device)

        accumulated_rows = run_iteration(
            prev_model, tokenizer, base_rows, accumulated_rows,
            cfg, iteration, iter_dir, summary_cb,
        )

    summary_cb.close()
    logger.info(f"Done. Final model saved at: "
                f"{os.path.join(cfg.checkpoints_dir, f'iter_{cfg.num_iterations - 1}')}")


if __name__ == "__main__":
    main()
