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
import shutil
import torch
import os

# Must be set before torch is imported so the CUDA allocator picks it up.
# expandable_segments: reduces fragmentation when many tensors of varying sizes
# are allocated/freed rapidly (typical during forward+backward of variable-length sequences).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Debug env variables
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
# os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")
# os.environ.setdefault("TORCH_LOGS", "+dynamo")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


# ── Setup ────────────────────────────────────────────────────────────────────

def setup(cfg):
    logger.info("=== setup() — creating output directories and saving config ===")
    logger.info(f"  output_dir:          {cfg.output_dir}")
    logger.info(f"  synthetic_cache_dir: {cfg.synthetic_cache_dir}")
    logger.info(f"  checkpoints_dir:     {cfg.checkpoints_dir}")
    logger.info(f"  tensorboard_dir:     {cfg.tensorboard_dir}")
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.synthetic_cache_dir)
    ensure_dir(cfg.checkpoints_dir)
    ensure_dir(cfg.tensorboard_dir)
    if cfg.enable_profiler:
        logger.info(f"  Profiler enabled — profile_dir: {cfg.profile_dir}")
        ensure_dir(cfg.profile_dir)
    config_path = os.path.join(cfg.output_dir, "config.json")
    save_json(config_path, asdict(cfg))
    logger.info(f"  Config snapshot written to: {config_path}")
    set_seed(cfg.seed)
    logger.info(f"  Random seed set to {cfg.seed} (affects weight init, data shuffling, sampling).")
    logger.info("setup() complete — directories created, config saved, seed fixed.")


def load_tokenizer_and_data(cfg):
    logger.info("=== load_tokenizer_and_data() — loading tokenizer and base dataset ===")
    tokenizer = load_tokenizer(cfg)
    logger.info(f"  Tokenizer loaded. vocab_size={tokenizer.vocab_size}, pad_token='{tokenizer.pad_token}'")

    data_source = cfg.data_path if cfg.data_path else f"{cfg.dataset_name} / split={cfg.train_split}"
    logger.info(f"  Loading base dataset from: {data_source}")
    logger.info(f"  max_data_load cap: {cfg.max_data_load if cfg.max_data_load > 0 else 'unlimited'}")
    base_ds = load_base_dataset_fixed(
        dataset_name=cfg.dataset_name if cfg.dataset_name else None,
        dataset_config_name=cfg.dataset_config_name if cfg.dataset_config_name else None,
        split=cfg.train_split,
        data_path=cfg.data_path if cfg.data_path else None,
        limit=cfg.max_data_load if cfg.max_data_load > 0 else None,
    )
    base_rows = [base_ds[i] for i in range(len(base_ds))]
    logger.info(
        f"  Base dataset materialised into {len(base_rows)} rows (Python list). "
        f"Each row has keys: {list(base_rows[0].keys()) if base_rows else 'N/A'}.")
    logger.info(
        f"load_tokenizer_and_data() complete — {len(base_rows)} training prompts available "
        f"(each iteration samples up to {cfg.synthetic_examples_per_iteration} of these).")
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


# ── Checkpoint cleanup ────────────────────────────────────────────────────────

def _cleanup_trainer_checkpoints(iter_dir: str):
    """Delete HF Trainer checkpoint-N subdirs after the merged model is fully saved.

    The merged model is written directly into iter_dir by trainer.save_model();
    the checkpoint-N subdirs are only needed for mid-training resume and can be
    removed once the .done sentinel confirms the iteration completed cleanly.
    """
    for entry in os.scandir(iter_dir):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            shutil.rmtree(entry.path)
            logger.info(f"Removed intermediate trainer checkpoint: {entry.path}")


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

def build_callbacks(cfg, iteration, summary_cb, tokenizer):
    logger.info(f"=== build_callbacks() — assembling trainer callbacks for iteration {iteration} ===")
    callbacks = (
        [TorchProfilerCallback(
            cfg, spin_iteration=iteration, tb_writer=summary_cb.writer)]
        if cfg.enable_profiler else []
    )
    if cfg.enable_profiler:
        logger.info("  [+] TorchProfilerCallback — records operator-level GPU/CPU timelines.")

    tb_log_dir = os.path.join(cfg.tensorboard_dir, f"iter_{iteration}")
    callbacks.append(
        TensorBoardCallbackExtended(
            log_dir=tb_log_dir,
            log_histograms=False,
            cfg=cfg,
            tokenizer=tokenizer,
            iteration=iteration,
        ),
    )
    logger.info(f"  [+] TensorBoardCallbackExtended — per-step metrics → {tb_log_dir}")

    if cfg.log_trainable_parameters:
        callbacks.append(
            TensorBoardParameterStatsCallback(
                cfg=cfg,
                run_name=f"spin_iter_{iteration}",
            )
        )
        logger.info("  [+] TensorBoardParameterStatsCallback — per-parameter weight/gradient histograms.")

    callbacks.append(MemoryProbeCallback(writer=summary_cb.writer))
    logger.info("  [+] MemoryProbeCallback — GPU/CPU memory logged to global TensorBoard writer.")

    callbacks.append(summary_cb)
    logger.info("  [+] SPINIterationSummaryCallback — cross-iteration summary (last, so it fires after all others).")
    logger.info(f"build_callbacks() complete — {len(callbacks)} callbacks registered.")
    return callbacks


def run_training(train_model, train_rows, ref_logprobs, tokenizer, cfg, iter_dir, iteration, summary_cb):
    """Build dataset, initialize trainer, train, merge LoRA, save checkpoint. Returns saved model."""
    logger.info(f"=== run_training() — SPIN iteration {iteration} ===")
    logger.info(f"  Input: {len(train_rows)} training rows, ref_logprobs present={ref_logprobs is not None}")

    logger.info("  Step 1/6: Pre-tokenizing all training rows into SPINDataset...")
    dataset = SPINDataset(train_rows, tokenizer, cfg,
                          ref_logprobs=ref_logprobs)
    n_train_rows = len(train_rows)
    del train_rows, ref_logprobs
    gc.collect()
    logger.info(f"  SPINDataset created: {len(dataset)} (chosen, rejected) pairs tokenised. "
                f"Raw text and ref_logprobs freed from memory.")

    collator = SPINDataCollator(tokenizer)
    logger.info("  Step 2/6: Computing training hyperparameters for this iteration...")
    spin_lambda = get_iteration_lambda(cfg, iteration)
    lr = get_iteration_lr(cfg, iteration)
    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    logger.info(f"  spin_lambda={spin_lambda}, lr={lr:.2e}, loss_type={cfg.loss_type}")
    logger.info(f"  per_device_batch={cfg.per_device_train_batch_size}, "
                f"grad_accum={cfg.gradient_accumulation_steps}, "
                f"effective_batch={eff_batch}, epochs={cfg.num_epochs_per_iteration}")

    args = build_training_args(cfg, iter_dir, lr, logging_dir=os.path.join(cfg.tensorboard_dir, f"iter_{iteration}"))
    logger.info(f"  TrainingArguments built. output_dir={iter_dir}, optimizer={cfg.optimizer}")

    trainer_cls = RMSPropSPINTrainer if cfg.optimizer.lower() == "rmsprop" else SPINTrainer
    logger.info(f"  Trainer class selected: {trainer_cls.__name__} (optimizer='{cfg.optimizer}')")

    logger.info("  Step 3/6: Setting up callbacks and registering iteration metadata...")
    summary_cb.set_iteration(iteration, spin_lambda=spin_lambda, dataset_size=n_train_rows)
    callbacks = build_callbacks(cfg, iteration, summary_cb, tokenizer)

    logger.info("  Step 4/6: Initialising trainer and moving model to device...")
    log_memory("before_trainer_init")
    trainer = trainer_cls(
        model=train_model,
        spin_lambda=spin_lambda,
        loss_type=cfg.loss_type,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    log_memory("after_trainer_init")

    resume_ckpt = get_last_checkpoint(iter_dir)
    if resume_ckpt:
        logger.info(f"  Mid-iteration resume detected — continuing from checkpoint: {resume_ckpt}")
    else:
        logger.info(f"  No mid-iteration checkpoint found in {iter_dir} — starting fresh.")

    logger.info("  Step 5/6: Starting trainer.train()...")
    log_memory("before_trainer_train")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    log_memory("after_trainer_train")
    logger.info(f"  trainer.train() finished for iteration {iteration}.")

    logger.info("  Step 6/6: Merging LoRA adapters and saving full model checkpoint...")
    train_model = merge_lora_and_get_base(train_model, cfg)
    trainer.model = train_model
    trainer.save_model(iter_dir)
    logger.info(f"  Model checkpoint saved to {iter_dir}. "
                f"(LoRA merged={cfg.use_lora}, plain AutoModelForCausalLM on disk)")
    logger.info(f"run_training() complete — iteration {iteration} model saved.")
    return train_model


# ── Iteration orchestration ───────────────────────────────────────────────────

def run_iteration(prev_model, tokenizer, base_rows, accumulated_rows, cfg, iteration, iter_dir, summary_cb):
    """Run one complete SPIN iteration. Returns updated accumulated_rows."""
    logger.info(f"=== run_iteration() — SPIN iteration {iteration} (5 phases) ===")

    # ── Phase 1: Generate or load synthetic (rejected) responses ──────────────
    logger.info(f"  Phase 1/5: Generating synthetic responses with frozen prev_model (π_prev)...")
    synthetic_rows, synth_cache_hit = get_or_generate_synthetic(
        prev_model, tokenizer, base_rows, cfg, iteration
    )
    logger.info(f"  Phase 1 done — {len(synthetic_rows)} synthetic rows "
                f"({'loaded from cache' if synth_cache_hit else 'freshly generated'}).")

    # ── Phase 2: Assemble training data ───────────────────────────────────────
    logger.info(f"  Phase 2/5: Assembling training dataset "
                f"(accumulate_previous_synthetic={cfg.accumulate_previous_synthetic})...")
    if cfg.accumulate_previous_synthetic:
        accumulated_rows = synthetic_rows if iteration == 0 else accumulated_rows + synthetic_rows
        train_rows = accumulated_rows
        logger.info(f"  Curriculum accumulation: {len(train_rows)} total rows "
                    f"(prior accumulated={len(accumulated_rows) - len(synthetic_rows)}, "
                    f"new={len(synthetic_rows)}).")
    else:
        train_rows = synthetic_rows
        logger.info(f"  Fixed-size training set: {len(train_rows)} rows (current iteration only).")
    del synthetic_rows

    # ── Phase 3: Score chosen+rejected under frozen prev_model ────────────────
    logger.info(f"  Phase 3/5: Computing reference log-probs under frozen π_prev "
                f"({len(train_rows)} rows to score)...")
    ref_logprobs = get_or_compute_ref_logprobs(
        prev_model, tokenizer, train_rows, cfg, iteration, synth_cache_hit
    )
    logger.info(f"  Phase 3 done — {len(ref_logprobs)} ref log-prob pairs ready.")

    # ── Phase 4: Convert prev_model → trainable π_θ and train ────────────────
    logger.info(f"  Phase 4/5: Converting frozen π_prev → trainable π_θ (use_lora={cfg.use_lora})...")
    train_model = make_trainable(prev_model, cfg)
    del prev_model
    logger.info("  prev_model reference released (memory eligible for GC).")

    train_model = run_training(
        train_model, train_rows, ref_logprobs, tokenizer, cfg, iter_dir, iteration, summary_cb
    )

    # ── Phase 5: Save and clean up ────────────────────────────────────────────
    logger.info(f"  Phase 5/5: Saving tokenizer and writing completion sentinel...")
    tokenizer.save_pretrained(iter_dir)
    logger.info(f"  Tokenizer saved to {iter_dir}.")

    done_path = os.path.join(iter_dir, ".done")
    open(done_path, "w").close()
    logger.info(f"  .done sentinel written — iteration {iteration} will be skipped on any future restart.")

    _cleanup_trainer_checkpoints(iter_dir)

    free_model(train_model)
    logger.info("  GPU/CPU memory freed (model deleted, GC collected, CUDA cache cleared).")
    logger.info(f"run_iteration() complete — iteration {iteration} finished successfully.")

    return accumulated_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║              SPIN Training — main() starting                 ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    cfg = parse_args()
    logger.info(f"Configuration parsed. Key settings:")
    logger.info(f"  model:              {cfg.model_name_or_path}")
    logger.info(f"  dataset:            {cfg.data_path or cfg.dataset_name} (split={cfg.train_split})")
    logger.info(f"  num_iterations:     {cfg.num_iterations}")
    logger.info(f"  epochs/iter:        {cfg.num_epochs_per_iteration}")
    logger.info(f"  synthetic/iter:     {cfg.synthetic_examples_per_iteration}")
    logger.info(f"  optimizer:          {cfg.optimizer}, lr={cfg.learning_rate:.2e}")
    logger.info(f"  use_lora:           {cfg.use_lora} (r={cfg.lora_r}, α={cfg.lora_alpha})")
    logger.info(f"  device:             {cfg.device}, dtype={cfg.torch_dtype}")
    logger.info(f"  loss_type:          {cfg.loss_type}, lambda_initial={cfg.lambda_initial}")
    logger.info(f"  compile_model:      {cfg.compile_model} ({cfg.compile_backend}, {cfg.compile_mode})")

    setup(cfg)

    tokenizer, base_rows = load_tokenizer_and_data(cfg)

    start_iteration = find_start_iteration(cfg)
    if start_iteration > 0:
        logger.info(f"Resuming: iterations 0–{start_iteration - 1} already complete. "
                    f"Starting at iteration {start_iteration}.")
    else:
        logger.info("Fresh run — starting from iteration 0.")

    accumulated_rows = restore_accumulated_rows(cfg, start_iteration)
    logger.info(f"Accumulated rows from prior iterations: {len(accumulated_rows)}")

    global_tb_dir = os.path.join(cfg.tensorboard_dir, "global")
    ensure_dir(global_tb_dir)
    summary_cb = SPINIterationSummaryCallback(log_dir=global_tb_dir, cfg=cfg)
    logger.info(f"Global TensorBoard writer initialised at: {global_tb_dir}")
    logger.info(f"Starting SPIN loop: iterations {start_iteration} → {cfg.num_iterations - 1}.")

    for iteration in range(start_iteration, cfg.num_iterations):
        remaining = cfg.num_iterations - iteration
        logger.info(f"")
        logger.info(f"╔══ SPIN ITERATION {iteration}/{cfg.num_iterations - 1} "
                    f"({remaining} remaining) ══════════════════════════════╗")
        iter_dir = os.path.join(cfg.checkpoints_dir, f"iter_{iteration}")
        ensure_dir(iter_dir)
        logger.info(f"  iter_dir: {iter_dir}")

        prev_model_path = (
            cfg.model_name_or_path if iteration == 0
            else os.path.join(cfg.checkpoints_dir, f"iter_{iteration - 1}")
        )
        logger.info(f"  Loading π_prev (frozen reference model) from: {prev_model_path}")
        log_memory(f"before_load_prev_model_iter{iteration}")
        prev_model = load_causal_lm(
            prev_model_path, cfg, trainable=False).to(cfg.device)
        log_memory(f"after_load_prev_model_iter{iteration}")

        accumulated_rows = run_iteration(
            prev_model, tokenizer, base_rows, accumulated_rows,
            cfg, iteration, iter_dir, summary_cb,
        )
        logger.info(f"╚══ SPIN ITERATION {iteration} COMPLETE ══════════════════════════════════╝")

    summary_cb.close()
    final_model_path = os.path.join(cfg.checkpoints_dir, f"iter_{cfg.num_iterations - 1}")
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║              SPIN Training — main() complete                 ║")
    logger.info(f"║  Final model: {final_model_path}")
    logger.info("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
