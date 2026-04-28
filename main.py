import os

# Must be set before torch is imported so the CUDA allocator picks it up.
# expandable_segments: reduces fragmentation when many tensors of varying sizes
# are allocated/freed rapidly (typical during forward+backward of variable-length sequences).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")



import gc
import random
import logging
from dataclasses import asdict
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from utils import *
from spin_dataset import *
from spin_data_collator import *
from trainer_callback import *
from spin_trainer import *



logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)



def main():
    """Entry point for SPIN (Self-Play Fine-Tuning) training.

    Orchestrates the outer SPIN loop:
      1. Parse config and set up output directories.
      2. Load the tokenizer and base dataset once (reused across all iterations).
      3. Detect the first incomplete iteration (supports crash resume via .done sentinels).
      4. For each iteration:
         a. Load π_prev (previous-iteration or base model, frozen).
         b. Generate or reload synthetic 'rejected' responses.
         c. Score every (chosen, rejected) pair under π_prev (ref log-probs).
         d. Convert π_prev to a trainable model (LoRA or full fine-tune).
         e. Train with SPINTrainer; log progress via TensorBoard callbacks.
         f. Merge LoRA adapters (if used), save model + tokenizer, write .done.
         g. Free GPU memory before the next iteration.
      5. Close the global cross-iteration TensorBoard writer.

    The final checkpoint is saved at cfg.checkpoints_dir/iter_{N-1}/ and is a
    plain AutoModelForCausalLM with no PEFT dependency.
    """
    cfg = parse_args()

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.synthetic_cache_dir)
    ensure_dir(cfg.checkpoints_dir)
    if cfg.enable_profiler:
        ensure_dir(cfg.profile_dir)

    save_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
    set_seed(cfg.seed)
    logger.info("============================ 1 ==================================")

    tokenizer = load_tokenizer(cfg)
    base_ds = load_base_dataset_fixed(
        dataset_name=cfg.dataset_name if cfg.dataset_name else None,
        dataset_config_name=cfg.dataset_config_name if cfg.dataset_config_name else None,
        split=cfg.train_split,
        data_path=cfg.data_path if cfg.data_path else None,
        limit=cfg.max_data_load if cfg.max_data_load > 0 else None,
    )

    base_rows = [base_ds[i] for i in range(len(base_ds))]
    logger.info(f"Loaded full dataset: {len(base_rows)} rows (sampled per-iteration).")

    # ── Resume detection ────────────────────────────────────────────────────
    # Find the first iteration that hasn't written its .done sentinel yet.
    # Completed iterations are skipped entirely; the loop picks up from there.
    start_iteration = find_start_iteration(cfg)
    if start_iteration > 0:
        logger.info(f"Resuming: iterations 0–{start_iteration - 1} already complete. "
                    f"Starting at iteration {start_iteration}.")

    # Restore accumulated synthetic rows from saved JSONL files so the growing
    # curriculum is intact even when resuming after a kill.
    accumulated_rows = []
    if start_iteration > 0 and cfg.accumulate_previous_synthetic and cfg.save_synthetic_jsonl:
        for prev_iter in range(start_iteration):
            prev_synth_path = os.path.join(cfg.synthetic_cache_dir, f"iter_{prev_iter}.jsonl")
            if os.path.exists(prev_synth_path):
                prev_rows = load_jsonl(prev_synth_path)
                accumulated_rows.extend(prev_rows)
                logger.info(f"  Restored {len(prev_rows)} rows from iter_{prev_iter}.jsonl "
                            f"(accumulated total: {len(accumulated_rows)}).")

    logger.info("============================ 2 ==================================")

    # Created once outside the loop so the x-axis spans all SPIN iterations.
    # Writes to tb_global/ so it shows up as a separate run in TensorBoard,
    # separate from the per-iteration tb_logs/ runs.
    global_tb_dir = os.path.join(cfg.output_dir, "tb_global")
    ensure_dir(global_tb_dir)
    summary_cb = SPINIterationSummaryCallback(log_dir=global_tb_dir, cfg=cfg)

    for iteration in range(start_iteration, cfg.num_iterations):
        logger.info(f"========== SPIN ITERATION {iteration} ==========")
        iter_dir = os.path.join(cfg.checkpoints_dir, f"iter_{iteration}")
        ensure_dir(iter_dir)

        if iteration == 0:
            prev_model_path = cfg.model_name_or_path
        else:
            prev_model_path = os.path.join(cfg.checkpoints_dir, f"iter_{iteration - 1}")

        logger.info(f"prev_model_path: {prev_model_path}")

        prev_model = load_causal_lm(prev_model_path, cfg, trainable=False).to(cfg.device)

        limit = cfg.synthetic_examples_per_iteration if cfg.synthetic_examples_per_iteration > 0 else len(base_rows)
        synth_path = os.path.join(cfg.synthetic_cache_dir, f"iter_{iteration}.jsonl")

        if cfg.save_synthetic_jsonl and os.path.exists(synth_path):
            # Reuse synthetic data from a previous (possibly interrupted) run of this
            # iteration. This is critical for correctness: if training was killed
            # mid-way and left a checkpoint, resuming requires the exact same dataset
            # that was used when the checkpoint was created. Regenerating with a freshly
            # shuffled base_rows would produce different rows and corrupt the resume.
            synthetic_rows = load_jsonl(synth_path)
            logger.info(f"Loaded {len(synthetic_rows)} cached synthetic rows from {synth_path} "
                        f"(skipping generation).")
            logger.info("============================ 3 (cached) ==================================")
            logger.info("============================ 4 (cached) ==================================")
        else:
            # Use a seed derived from (global seed + iteration) so the sample is
            # different each iteration but fully reproducible on resume.
            iter_rng = random.Random(cfg.seed + iteration)
            rows_for_generation = iter_rng.sample(base_rows, min(limit, len(base_rows)))
            logger.info(f"============================ 2.6 Length of dataset {len(rows_for_generation)}")
            logger.info("============================ 3 ==================================")
            synthetic_rows = generate_synthetic_responses(prev_model, tokenizer, rows_for_generation, cfg)
            del rows_for_generation
            logger.info("============================ 4 ==================================")
            if cfg.save_synthetic_jsonl:
                save_jsonl(synth_path, synthetic_rows)

        if cfg.accumulate_previous_synthetic:
            accumulated_rows = synthetic_rows if iteration == 0 else accumulated_rows + synthetic_rows
            train_rows = accumulated_rows
        else:
            train_rows = synthetic_rows
        del synthetic_rows
        logger.info("============================ 5 ==================================")

        # Compute ref logprobs while prev_model is frozen, then repurpose it as train_model
        ref_logprobs = compute_ref_logprobs(prev_model, tokenizer, train_rows, cfg)
        train_model = make_trainable(prev_model, cfg)
        del prev_model

        # Pre-tokenizes all rows in __init__; raw text no longer needed after this
        dataset = SPINDataset(train_rows, tokenizer, cfg, ref_logprobs=ref_logprobs)
        if not cfg.accumulate_previous_synthetic:
            pass  # accumulated_rows not used; nothing to clear
        gc.collect()
        collator = SPINDataCollator(tokenizer)
        spin_lambda = get_iteration_lambda(cfg, iteration)
        lr = get_iteration_lr(cfg, iteration)
        args = build_training_args(cfg, iter_dir, lr)
        logger.info("============================ 6 ==================================")

        trainer_cls = RMSPropSPINTrainer if cfg.optimizer.lower() == "rmsprop" else SPINTrainer
        summary_cb.set_iteration(iteration, spin_lambda=spin_lambda, dataset_size=len(train_rows))
        
        del train_rows, ref_logprobs

        callbacks = (
            [TorchProfilerCallback(cfg, spin_iteration=iteration, tb_writer=summary_cb.writer)]
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
        print(len(dataset), dataset)
        logger.info("============================ 6.5 ==================================")

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
        logger.info("============================ 7 ==================================")

        # Always resume from the latest HuggingFace checkpoint inside iter_dir if one exists.
        # On the first run of an iteration iter_dir has no checkpoint subdirs, so this is None.
        # On a restart after a kill mid-iteration, the trainer picks up where it left off.
        resume_ckpt = get_last_checkpoint(iter_dir)
        if resume_ckpt:
            logger.info(f"Resuming mid-iteration from checkpoint: {resume_ckpt}")
        logger.info("============================ 8 ==================================")

        log_memory("before_trainer_train")
        trainer.train(resume_from_checkpoint=resume_ckpt)
        logger.info("============================ 9 ==================================")

        # If LoRA was used, merge adapters into the base weights before saving.
        # This produces a plain AutoModelForCausalLM checkpoint with no PEFT
        # dependency so the next iteration's load_causal_lm() just works.
        train_model = merge_lora_and_get_base(train_model, cfg)
        trainer.model = train_model  # ensure trainer saves the unwrapped base model, not the PEFT wrapper
        trainer.save_model(iter_dir)
        logger.info("============================ 10 ==================================")

        tokenizer.save_pretrained(iter_dir)
        logger.info("============================ 11 ==================================")

        # Write a sentinel so find_start_iteration() can skip this iteration on restart.
        # Written AFTER both model and tokenizer are fully saved.
        open(os.path.join(iter_dir, ".done"), "w").close()
        logger.info(f"Iteration {iteration} complete. Sentinel written to {iter_dir}/.done")

        free_model(train_model)
        logger.info("============================ 12 ==================================")

    summary_cb.close()
    logger.info(f"Done. Final model saved at: {os.path.join(cfg.checkpoints_dir, f'iter_{cfg.num_iterations - 1}')}")


if __name__ == "__main__":
    main()