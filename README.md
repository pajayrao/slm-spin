# slm-spin

Implementation of the SPIN (Self-Play Fine-Tuning) algorithm for Small Language Models, optimised for consumer and research GPUs.

---

## Table of Contents

1. [What is SPIN?](#what-is-spin)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
   - [Tested and Planned Models](#tested-and-planned-models)
   - [Local Dataset](#local-dataset)
   - [Running Evaluation](#running-evaluation)
4. [Configuration Reference](#configuration-reference)
   - [SPIN Loss](#spin-loss)
5. [Memory Tips for Small GPUs](#memory-tips-for-small-gpus--16-gb)
6. [Output Structure](#output-structure)
7. [Resuming After a Crash](#resuming-after-a-crash)
8. [TensorBoard](#tensorboard-training)
9. [Project Structure](#project-structure)
10. [Implementation Deep-Dive](#implementation-deep-dive)
    - [Outer Loop — main.py](#outer-loop--mainpy)
    - [Configuration — spin_config.py](#configuration--spin_configpy)
    - [Dataset — spin_dataset.py](#dataset--spin_datasetpy)
    - [Data Collator — spin_data_collator.py](#data-collator--spin_data_collatorpy)
    - [Trainer and Loss — spin_trainer.py](#trainer-and-loss--spin_trainerpy)
    - [Utilities — utils.py](#utilities--utilspy)
    - [Callbacks — trainer_callback/](#callbacks--trainer_callback)
11. [Evaluation — evaluate.py](#evaluation--evaluatepy)
    - [Programmatic vs Standalone Usage](#programmatic-vs-standalone-usage)
    - [Benchmark Descriptions](#benchmark-descriptions)
    - [Scoring Strategy](#scoring-strategy)
    - [Per-Task Details](#per-task-details)
    - [Evaluation Loop and Output](#evaluation-loop-and-output)
12. [Detailed Explanation](#detailed-explanation)
    - [The Game-Theoretic View](#the-game-theoretic-view)
    - [The Mathematical Formulation](#the-mathematical-formulation)
    - [Why Does This Work?](#why-does-this-work)
13. [How SPIN Works — Step by Step](#how-spin-works--step-by-step)
    - [The Inner Batch Loop](#the-inner-batch-loop)
    - [Dataset Order](#dataset-order)
    - [Iteration Completion](#iteration-completion)
    - [Crash-Safe Resumption](#crash-safe-resumption)
14. [SPIN vs Standard Fine-Tuning](#spin-vs-standard-fine-tuning)
    - [Key Advantage: Self-Improvement Without New Data](#key-advantage-self-improvement-without-new-data)
    - [Key Advantage: Reference Regularization](#key-advantage-reference-regularization)
    - [Key Limitation vs RLHF/DPO](#key-limitation-vs-rlhfdpo)
    - [Other LLM Training Techniques](#other-llm-training-techniques)
      - [Pre-training](#pre-training)
      - [Continued Pre-training (CPT)](#continued-pre-training-cpt)
      - [Supervised Fine-Tuning (SFT) and Instruction Tuning](#supervised-fine-tuning-sft-and-instruction-tuning)
      - [RLHF + PPO](#rlhf--ppo)
      - [DPO — Direct Preference Optimization](#dpo--direct-preference-optimization)
      - [IPO — Identity Preference Optimization](#ipo--identity-preference-optimization)
      - [KTO — Kahneman-Tversky Optimization](#kto--kahneman-tversky-optimization)
      - [ORPO — Odds Ratio Preference Optimization](#orpo--odds-ratio-preference-optimization)
      - [SimPO — Simple Preference Optimization](#simpo--simple-preference-optimization)
      - [GRPO — Group Relative Policy Optimization](#grpo--group-relative-policy-optimization)
      - [RLAIF — Reinforcement Learning from AI Feedback](#rlaif--reinforcement-learning-from-ai-feedback)
      - [Constitutional AI (CAI)](#constitutional-ai-cai)
      - [Knowledge Distillation](#knowledge-distillation)
      - [Technique Selection Guide](#technique-selection-guide)

---

## What is SPIN?

SPIN — **Self-Play Fine-Tuning** — is a method for progressively aligning a language
model using only the dataset it was originally trained on — **no new human labels, no
reward model, no preference data.** It was introduced in *"Self-Play Fine-Tuning Converts
Weak Language Models to Strong Language Models"* (Chen et al., 2024).

**The problem it solves.** Standard supervised fine-tuning (SFT) teaches a model to
assign high probability to human-written responses. But it does not explicitly teach
the model to prefer those responses *over its own outputs*. A model trained with SFT
might assign similarly high probability to the human response *and* to its own vague
or repetitive completions of the same prompt — the training signal contains no contrast
between good and bad outputs.

**The SPIN mechanism** introduces that contrast through a four-step loop:

1. Freeze a snapshot of the current model (the **opponent**, `π_prev`).
2. Use `π_prev` to generate a **synthetic response** for every training prompt — these
   become the *rejected* side of each training pair.
3. Train the live model (`π_θ`) to assign higher log-probability to the **human**
   response than to `π_prev`'s synthetic response — scored *relative* to where `π_prev`
   started, so the model is rewarded for improving beyond its prior self.
4. Save the trained `π_θ` as the new `π_prev` and repeat.

Each round the synthetic responses become harder to beat because the opponent is
stronger. This self-competition drives the model toward the human data distribution
without ever requiring new annotations.

## Installation

```bash
pip install -r requirements.txt
```

Key runtime dependencies: `transformers`, `peft`, `torch`, `datasets`, `accelerate`,
`tensorboard`.

A CUDA-capable GPU is required for practical training. The code targets CUDA 13.0 /
PyTorch 2.11.

---

## Quick Start

```bash
python main.py \
  --model_name_or_path distilbert/distilgpt2 \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --train_split train_sft \
  --num_iterations 3 \
  --data_batch_size 65536 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --output_dir ./runs/my_run
```

All `SPINConfig` fields are exposed as CLI flags — pass any field as `--field_name value`.

### Tested and Planned Models

This implementation is being systematically evaluated on the following models in size order:

| Model | Params | Status |
|-------|--------|--------|
| `roneneldan/TinyStories-Instruct-33M` | 33M | No significant gain as model is too small. |
| `distilbert/distilgpt2` | 82M | Model is not fine tuned for instruct dataset. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | Spin Play ongoing. Required multiple rounds of hyperparameter tuning. |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | Queued |
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | Queued |
| `google/gemma-3-1b-it` | 1B | Queued |

LoRA target modules are auto-detected per architecture — no config change needed when switching models.

### Local dataset

```bash
python main.py \
  --data_path ./my_data.jsonl \
  --model_name_or_path /path/to/local/model \
  --num_iterations 3
```

JSONL files must contain records with `prompt` and `response` keys, **or** a `messages`
list of `{"role": ..., "content": ...}` dicts.

### Running Evaluation

All CLI defaults are derived from `SPINConfig()`, so they automatically align with the
training output layout (`./spin_outputs/checkpoints`, `./spin_outputs/eval_results`,
`./spin_outputs/tensorboard/eval_compare`).

```bash
# Evaluate all iter_* checkpoints in the default directory (./spin_outputs/checkpoints)
python evaluate.py

# Evaluate a specific subset of iterations
python evaluate.py --iters iter_0 iter_2 iter_4

# Smoke test — limit examples per task (do NOT use for real benchmarks)
python evaluate.py --limit 50

# Override shot counts for specific tasks
python evaluate.py --n-shots arc_challenge=10 gsm8k=3

# Run only a subset of tasks
python evaluate.py --tasks arc_challenge winogrande mmlu

# Exclude specific tasks (complement of --tasks; mutually exclusive with it)
python evaluate.py --skip-tasks hellaswag mmlu

# Custom checkpoint and output directories
python evaluate.py \
  --checkpoints-dir ./runs/my_run/checkpoints \
  --output-dir      ./runs/my_run/eval_results \
  --tensorboard-dir ./runs/my_run/tensorboard/eval

# Skip already-evaluated iterations (default); force re-evaluation
python evaluate.py --no-cache
```

---

## Configuration Reference

All options live in [spin_config.py](spin_config.py). The most important ones:

| Flag | Default | Description |
|------|---------|-------------|
| `model_name_or_path` | `distilbert/distilgpt2` | HuggingFace Hub ID or local checkpoint path |
| `dataset_name` | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset (overridden by `data_path`) |
| `num_iterations` | `5` | Number of SPIN outer loops |
| `data_batch_size` | `65536` | Rows per checkpoint batch — each batch runs all 3 steps atomically; smaller = more frequent crash-recovery saves |
| `lambda_initial` | `0.1` | SPIN loss scale λ for all but the last iteration |
| `lambda_final_iteration` | `5.0` | λ for the final iteration (stronger alignment push) |
| `loss_type` | `logistic` | `logistic` \| `hinge` \| `correlation` \| `exponential` |
| `use_lora` | `True` | Enable LoRA (strongly recommended on small GPUs) |
| `lora_r` | `16` | LoRA rank |
| `optimizer` | `rmsprop` | `rmsprop` (less memory) \| `adamw` |
| `per_device_train_batch_size` | `16` | Reduce to `1` on an 8 GB GPU |
| `gradient_accumulation_steps` | `32` | Compensate for small batch size. Must satisfy `data_batch_size ≥ per_device_train_batch_size × gradient_accumulation_steps` |
| `learning_rate` | `5e-5` | Peak LR for early iterations |
| `learning_rate_late` | `1e-5` | LR from `late_lr_start_iteration` onward |
| `max_length` | `512` | Max tokens (prompt + response) during training |
| `bf16` | `True` | bfloat16 mixed precision (Ampere+ GPU required) |
| `gradient_checkpointing` | `False` | Recompute activations to save ~10× memory at ~33% compute cost |
| `output_dir` | `./spin_outputs` | Root directory for all training outputs |
| `eval_output_dir` | `./spin_outputs/eval_results` | Directory for per-iteration JSON score files and the comparative summary |
| `eval_tensorboard_dir` | `./spin_outputs/tensorboard/eval_compare` | TensorBoard log directory for evaluation metrics |
| `eval_limit` | `None` | Max examples per task during evaluation — `None` = full dataset; set a small integer for a smoke test |
| `eval_batch_size` | `8` | GPU forward-pass batch size for log-likelihood scoring during evaluation |
| `eval_max_seq_len` | `2048` | Maximum token length (context + continuation) fed to the model during evaluation |
| `eval_gsm8k_max_new_tokens` | `256` | Maximum new tokens generated per response in the GSM8k benchmark |
| `eval_no_cache` | `False` | Re-evaluate iterations even if a cached `.parsed.json` result exists |
| `eval_run_after_training` | `True` | Automatically run benchmark evaluation right after each SPIN iteration's checkpoint is saved (not just at the end) |

### SPIN Loss

The loss operates on a margin per training example:

```
margin = λ × [(log π_θ(chosen) − log π_ref(chosen)) − (log π_θ(rejected) − log π_ref(rejected))]
```

A positive margin means the model has improved more on the human response than on the
synthetic one. The `loss_type` maps this margin to a scalar:

| `loss_type` | Formula | Notes |
|---|---|---|
| `logistic` | `softplus(−margin)` | Smooth, never saturates — recommended |
| `hinge` | `relu(1 − margin)` | Zero loss once margin > 1 |
| `correlation` | `1 − margin` | Constant gradient, easiest to tune |
| `exponential` | `exp(−margin)` | Aggressive on negative margins; can be unstable |

---

## Memory Tips for Small GPUs (≤ 16 GB)

- Set `--use_lora True` (default) — cuts gradient/optimizer memory by ~10–100×.
- Lower `--per_device_train_batch_size 1` and raise `--gradient_accumulation_steps 64`.
- Enable `--gradient_checkpointing True` — trades ~33% compute for ~10× less activation memory.
- Use `--optimizer rmsprop` — saves ~2 GB vs AdamW on a 1 B-parameter model.
- Reduce `--max_length 512` and `--max_prompt_length 256`.
- Set `--generation_batch_size 4` — each beam holds its own KV cache during generation.
- Set `--bf16 True` (default on Ampere+) or `--fp16 True` on older GPUs.

---

## Output Structure

```
spin_outputs/
├── config.json                           # Snapshot of SPINConfig for this run
├── synthetic/
│   ├── iter_0_batch_000000_synth.jsonl    # Step 1 output: synthetic responses for batch 0
│   ├── iter_0_batch_000000_logprobs.jsonl # Step 2 output: ref log-probs for batch 0
│   ├── iter_0_batch_000000_tokenized.pt   # Step 3 cache: pre-tokenized dataset for batch 0
│   ├── iter_0_batch_000001_synth.jsonl
│   ├── iter_0_batch_000001_logprobs.jsonl
│   ├── iter_0_batch_000001_tokenized.pt
│   └── ...
├── checkpoints/
│   ├── iter_0/
│   │   ├── batch_000000/                 # Step 3 output: merged model after batch 0
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── .done                     # Sentinel: all 3 steps done for this batch
│   │   ├── batch_000001/
│   │   │   └── ...
│   │   ├── config.json                   # Final iteration model (copy of last batch)
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── .done                         # Iteration-level sentinel
│   └── iter_1/
│       └── ...
├── eval_results/
│   ├── iter_0.parsed.json
│   ├── comparative_summary.txt
│   └── comparative_summary.json
└── tensorboard/
    ├── global/                           # Memory + cross-iteration signals
    ├── iter_0/
    │   ├── batch_000000/                 # Per-batch TensorBoard logs
    │   ├── batch_000001/
    │   └── ...
    ├── param_stats/
    ├── profile/
    └── eval_compare/
```

**Disk usage note:** Each batch checkpoint in `checkpoints/iter_i/batch_k/` holds a
full merged model. For a 270 M parameter model in bfloat16 this is ~540 MB per batch.
With many small batches this can grow large. Delete old batch checkpoints after
confirming the iteration completed (the iteration-level `.done` is the safe signal).

The final model is at `checkpoints/iter_{num_iterations-1}/`.

---

## Resuming After a Crash

No flags are needed. On restart, the script automatically resumes at the finest
granularity possible:

**Iteration level** — `find_start_iteration()` scans `checkpoints/iter_*/`
in reverse for an iteration-level `.done` sentinel. Completed iterations are skipped
entirely.

**Batch level** — within the current iteration, `_find_start_batch()` scans batches
0 → N in order and returns the first batch where any step is incomplete. Each step
is re-checked individually:

| What's missing | Action |
|---|---|
| `_synth.jsonl` absent or empty | Re-run Step 1 (synthetic generation) |
| `_logprobs.jsonl` absent or empty | Re-run Step 2 (logprob scoring) |
| batch `.done` absent | Re-run Step 3 (training); within training, `get_last_checkpoint()` resumes from a partial HF Trainer checkpoint if one exists |

Because each step writes its output atomically (write-to-temp → rename), a kill
mid-write leaves the previous valid file intact — no corruption, no re-running
earlier steps.

---

## TensorBoard (training)

```bash
tensorboard --logdir ./spin_outputs/tensorboard
```

The `global/` run plots metrics across all iterations on a single x-axis. Individual
`iter_N/` runs show per-step detail for each iteration.

The `global/` run plots memory metrics. Per-batch training metrics land under
`iter_N/batch_K/` — each data batch gets its own TensorBoard sub-run.

Available panels (depending on config flags):

- **Scalars** — loss, margin mean/std, win rate, log-probs, KL from ref, learning rate
- **PR Curves** — alignment accuracy per epoch
- **Projector** — token embedding shift across iterations (PCA / UMAP / t-SNE)
- **Histograms** — per-layer weight and gradient distributions
- **Trace** and **Memory** — PyTorch profiler output

### Logged Values Reference

Values are organised by the TensorBoard sub-run they appear in. Open `tensorboard --logdir ./spin_outputs/tensorboard` and select runs in the left panel to compare.

> Per-layer parameter/gradient histograms and per-parameter scalar stats (`parameters/*`, `parameter_delta/*`, `gradients/*`, `layer_health/*`) are written by `TensorBoardParameterStatsCallback` under `param_stats/` — they are excluded from the tables below because they repeat for every trainable weight tensor.

The **Trend** column shows the expected direction when comparing the same metric at a consistent point in time (e.g. end of training) **across SPIN iterations 0 → 1 → 2 → …**. For `spin_progress/*` the trend applies to the per-iteration value directly.

---

#### `train/*` — per optimizer step (`iter_N/batch_K/`)

Logged by `TensorBoardCallbackExtended` every time `SPINTrainer` calls `self.log()`, plus derived values computed in `on_step_end`.

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `train/loss` | SPIN loss for the current step | Loss function (e.g. `softplus(−margin)`) applied to the batch-mean margin | Starts at ≈ 0.693 when margin = 0; falls toward 0 as alignment improves within the iteration | ↓ decreases |
| `train/margin_mean` | Mean per-example margin across the batch | `mean(λ × ((log π_θ(chosen) − ref_chosen_logp) − (log π_θ(rejected) − ref_rejected_logp)))` | Core alignment signal. Starts at 0 at step 0 of each iteration (π_θ is a copy of π_prev); trends positive during training; resets to 0 at every iteration boundary | ↑→ rises, plateaus |
| `train/margin_std` | Standard deviation of per-example margins | `std(...)` of the same per-example margins | High std = uneven alignment across the batch. Narrows as training consistently aligns most examples | ↓→ falls, stabilises |
| `train/win_rate` | Fraction of batch examples where margin > 0 | `mean(margin > 0)` | Proportion of examples where π_θ is ahead of π_prev. Trends from ≈ 0.5 to 1.0 within an iteration | ↑→ rises toward 1.0 |
| `train/pi_chosen_logp` | Log-prob π_θ assigns to the human response | `(Σ_t log π_θ(token_t \| prompt, prior_tokens)) / n_response_tokens` — per-token average over response tokens (prompt tokens masked) | Becomes less negative as the model learns to predict human text more confidently. Values are per-token averages (typically −1 to −5), not raw sums | ↑ increases (less negative) |
| `train/pi_rejected_logp` | Log-prob π_θ assigns to the synthetic response | Same per-token average over synthetic response tokens | Should stay flat or drift downward while `pi_chosen_logp` rises. Both rising equally = model inflating all probabilities (SPIN margin catches this) | ↑ increases (synthetic quality improves each iteration) |
| `train/ref_chosen_logp` | Log-prob π_prev (frozen) assigns to the human response | Pre-computed in Step 2; loaded from `_logprobs.jsonl`; **constant** throughout the iteration. Per-token average (same normalization as `pi_chosen_logp`) | Fixed anchor for the chosen side. In a healthy run this becomes less negative iteration-to-iteration as each new π_prev is better aligned. **If it trends downward (more negative) across iterations the previous iteration's training made π_prev worse at human text — reduce LR or λ** | ↑ increases (less negative); ↓ = training is diverging |
| `train/ref_rejected_logp` | Log-prob π_prev (frozen) assigns to the synthetic response | Pre-computed in Step 2; **constant** throughout the iteration. Per-token average | Fixed anchor for the rejected side. Also rises as synthetic quality improves; the gap `ref_chosen − ref_rejected` narrows each iteration. Trending downward alongside `ref_chosen_logp` confirms model collapse | ↑ increases (less negative); ↓ = training is diverging |
| `train/logp_gap` | Log-prob gap chosen − rejected under π_θ | `pi_chosen_logp − pi_rejected_logp` (derived each logged step) | Should be positive and **growing** — the model is widening its preference for human over synthetic. Negative = clear misalignment | ↑ increases during active training |
| `train/kl_from_ref` | Signed average policy advantage vs π_prev (**not** a true KL divergence — can be negative) | `(mean(pi_chosen − ref_chosen) + mean(pi_rejected − ref_rejected)) / 2` — average log-ratio across **both** chosen and rejected sides | 0 at step 0 (π_θ identical to π_prev). **Positive** = model is improving on both sides. **Negative** = model is regressing on chosen or aggressively pushing down rejected without lifting chosen — a negative trend is an alignment alarm, not normal. Distinct from margin: margin can be positive even when kl_from_ref is negative (if rejected drops faster than chosen) | **→ near 0 is healthy**; sustained negative = check LR/λ |
| `train/spin_lambda` | λ scaling factor for this iteration | `lambda_initial` for all but the last iteration; `lambda_final_iteration` for the last | Confirms the two-phase λ schedule. The jump on the final iteration should visibly increase margins | →↑ stable, then jumps on the last iteration |
| `train/learning_rate` | Current LR at this step | Read from optimizer param group after scheduler step | Verify the two-phase LR: `learning_rate` early, dropping to `learning_rate_late` at `late_lr_start_iteration` | →↓ stable, then drops |
| `train/grad_global_norm` | L2 norm of all parameter gradients | `sqrt(Σ ‖g_i‖²)` across all parameters with a gradient — computed in `SPINTrainer.training_step` immediately after `loss.backward()` while gradients are alive, then logged via `self.log()`. HF Trainer calls `model.zero_grad()` before `on_step_end`, so this cannot be read reliably in a callback | Gradient explosion detector. Sustained spikes (> 10×) indicate LR or λ is too large | ↓→ falls, stabilises |
| `train/weight_global_norm` | L2 norm of all trainable parameters | `sqrt(Σ ‖w_i‖²)` across all `requires_grad=True` parameters — computed on GPU every step | Sanity check on model scale; should be broadly stable across updates | → stable |
| `train/weight_drift` | Weight movement since the start of this training call | `sqrt(Σ ‖w_i − w_initial‖²)` from the snapshot taken at `on_train_begin`, covering all trainable parameters. Logged every `parameter_log_interval` steps (throttled to avoid repeated CPU transfers) | Grows monotonically within a training call. Comparable magnitude across iterations = consistent update pressure | → similar magnitude each iteration |
| `train/throughput_sps` | Training samples processed per second | `per_device_train_batch_size / elapsed_step_time` | GPU utilisation proxy. Drops indicate I/O stalls, memory pressure, or kernel launch overhead | → stable |
| `train/perplexity` | Perplexity of the SPIN loss | `exp(min(loss, 20))` — capped to avoid overflow | More interpretable than raw loss. Falls from ≈ 2.0 (when loss = log 2) toward 1.0 as alignment improves | ↓→ falls, plateaus near 1.0 |
| `train/alignment_accuracy` | Binary step-level alignment indicator | `1.0` if `margin_mean > 0`, else `0.0` | Is the model ahead of the reference this step? Noisy — use `epoch/win_rate_mean` for a smoother view | ↑→ rises toward 1.0 |

The following are **auto-logged by HuggingFace Trainer** and appear in TensorBoard because `on_log` writes every key in the Trainer's `logs` dict under `train/`. They are not emitted by SPINTrainer directly.

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `train/epoch` | Fractional epoch progress | Computed by HF Trainer: `completed_steps / steps_per_epoch` | Cycles 0→1 over each batch's training. Useful as an x-axis proxy when steps and epochs are both present | → cycles 0→1 each training call |
| `train/grad_norm` | Gradient L2 norm before clipping | Computed by HF Trainer during gradient clipping, before `optimizer.step()` and `zero_grad()` — always valid. Distinct from `train/grad_global_norm` which is computed by SPINTrainer after `loss.backward()` and includes all params (not only clipped) | Logged every step. Use alongside `grad_global_norm` to catch gradient spikes | ↓→ falls, stabilises |
| `train/total_flops` | Accumulated FLOPs for this training run | Estimated by HF Trainer from model architecture; **not all architectures are supported** — unsupported models log 0 or near-zero | Rough compute cost estimate. Values near 0 or erratic indicate HF Trainer could not infer FLOPs for this model family | → stable per run (accumulated total) |
| `train/train_loss` | Average SPIN loss over the entire training run | `accumulated_loss / total_steps` — logged once at the end of training by HF Trainer | Per-run average loss (single dot per run in TensorBoard). Complements `iteration_summary/final_loss` which is the last-step loss | ↓ decreases across iterations |
| `train/train_runtime` | Total wall-clock time for this training run (seconds) | Logged once by HF Trainer at `on_train_end` | Per-run training time. Significant variation between runs indicates data loading or GPU memory issues | → stable |
| `train/train_samples_per_second` | Overall throughput for the training run | `total_samples / train_runtime` — logged once at `on_train_end` | Run-level average throughput (vs `train/throughput_sps` which is per-step). Use to compare across iterations | → stable |
| `train/train_steps_per_second` | Overall steps-per-second for the training run | `total_steps / train_runtime` — logged once at `on_train_end` | Similar to `train_samples_per_second` but in optimizer steps. Both are scatter-plot style (one dot per run) | → stable |

---

#### `system/*` — per optimizer step (`iter_N/batch_K/`)

Logged by `TensorBoardCallbackExtended` (`on_step_end`) and `MemoryProbeCallback` (first 3 steps always; then every `log_every_n_steps`).

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `system/gpu_alloc_mb` | GPU memory actively holding tensor data | `torch.cuda.memory_allocated() / 1024²` | Active footprint of weights, activations, and optimizer states. Persistent jumps at batch boundaries indicate allocation leaks | → stable |
| `system/gpu_reserved_mb` | GPU memory held by PyTorch's caching allocator | `torch.cuda.memory_reserved() / 1024²` | Always ≥ `gpu_alloc_mb`. The gap (reserved − allocated) is the allocator's free list. Unbounded growth = call `torch.cuda.empty_cache()` | → stable |
| `system/gpu_util_pct` | GPU compute utilisation (%) | `pynvml.nvmlDeviceGetUtilizationRates(handle).gpu`; falls back to `torch.cuda.utilization()` | Values < 50% during training indicate CPU-GPU pipeline stalls or a batch size that is too small | → stable (ideally high) |
| `system/cpu_rss_mb` | Process RSS in CPU RAM | `psutil.Process(os.getpid()).memory_info().rss / 1024²` | Should stay stable across batches. Steady growth across iterations = dataset object or tensor not freed between iterations | → stable |

---

#### `epoch/*` — per epoch (`iter_N/batch_K/`)

Logged by `TensorBoardCallbackExtended` at `on_epoch_end`. X-axis is epoch number within one data batch's training.

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `epoch/loss_mean` | Average loss over all steps in the epoch | `mean(step losses this epoch)` | Smoother loss view. Should fall epoch-over-epoch within each batch's training | ↓ decreases |
| `epoch/loss_min` | Best loss seen in the epoch | `min(step losses this epoch)` | The peak alignment reached at any point in the epoch | ↓ decreases |
| `epoch/loss_max` | Worst loss seen in the epoch | `max(step losses this epoch)` | Large gap between `loss_min` and `loss_max` = high step-to-step variance | ↓ decreases |
| `epoch/loss_final` | Loss at the last step of the epoch | Last step loss before reset | Inherited by the next epoch; directly comparable epoch-to-epoch | ↓ decreases |
| `epoch/margin_mean` | Average SPIN margin over the epoch | `mean(margin_mean values this epoch)` | Epoch-level alignment health; should rise each epoch | ↑ increases |
| `epoch/margin_final` | Margin at the last step of the epoch | Last step margin before reset | Alignment state at the epoch boundary | ↑ increases |
| `epoch/win_rate_mean` | Average win rate over the epoch | `mean(win_rate values this epoch)` | More stable than per-step win rate; stagnation = plateau | ↑→ rises toward 1.0 |
| `epoch/win_rate_final` | Win rate at the last step of the epoch | Last step win rate before reset | Win rate at the epoch boundary; stagnation epoch-to-epoch signals a plateau | ↑→ rises toward 1.0 |
| `epoch/logp_gap_mean` | Average log-prob gap (chosen − rejected) over the epoch | `mean(pi_chosen_logp − pi_rejected_logp per step)` | Negative mean = model prefers synthetic over human; should be positive and growing | ↑ increases |
| `epoch/logp_gap_final` | Log-prob gap at the last step of the epoch | Last step gap before reset | Trailing indicator; use alongside `logp_gap_mean` | ↑ increases |

> `epoch/loss_distribution` is also written as a **histogram** of all step losses in the epoch (visible in the Histograms tab). The distribution should shift toward 0 across iterations.
>
> `train/alignment_pr_curve` is written at `on_epoch_end` if `log_pr_curves=True` — see Non-scalar outputs below.

---

#### `iteration_summary/*` — end of each batch's training phase (`iter_N/batch_K/`)

Logged by `TensorBoardCallbackExtended` at `on_train_end`. Written once per `_step_train()` call (once per data batch).

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `iteration_summary/total_weight_drift` | Total parameter movement from start to end of this batch's training | `sqrt(Σ ‖w_final − w_initial‖²)` across all trainable parameters | How much the model changed in this batch. Comparable magnitude across batches = consistent update pressure | → similar magnitude each iteration |
| `iteration_summary/final_loss` | SPIN loss at the very last step-level log entry | Taken from the last `state.log_history` entry that contains `margin_mean` (the SPIN step-level entry — the HF Trainer timing summary that is appended after it is deliberately skipped) | Loss value this batch hands forward; compare across iterations to confirm steady alignment | ↓ decreases |
| `iteration_summary/final_margin_mean` | Margin at the very last logged step | Last metrics entry | Final alignment state after all epochs on this data batch | ↑→ rises, plateaus |
| `iteration_summary/final_pi_chosen_logp` | π_θ log-prob on human response at the last step | Last metrics entry | How confidently the final batch model predicts human text | ↑ increases (less negative) |
| `iteration_summary/final_pi_rejected_logp` | π_θ log-prob on synthetic response at the last step | Last metrics entry | Should remain lower than `final_pi_chosen_logp`; rises as synthetic quality improves across iterations | ↑ increases (less negative) |
| `iteration_summary/final_win_rate` | Win rate at the very last logged step | Last metrics entry | Proportion of examples the model gets right at batch end | ↑→ rises toward 1.0 |
| `iteration_summary/final_kl_from_ref` | Signed average policy advantage vs π_prev at the last step | `(mean(pi_chosen − ref_chosen) + mean(pi_rejected − ref_rejected)) / 2` | **Positive** = model improved on both sides. **Negative** = model regressed on chosen or pushed rejected down harder than it lifted chosen. A consistently negative trend across batches is an alignment alarm | → near 0; negative = check LR/λ |
| `iteration_summary/final_logp_gap` | Log-prob gap at the last step | `final_pi_chosen_logp − final_pi_rejected_logp` | Should be positive and growing — model actively prefers human over synthetic. Only narrows toward 0 at Nash equilibrium | ↑ increases during active training |

---

#### `hparam/*` — HParams tab (`iter_N/batch_K/hparams/`)

Logged by `TensorBoardCallbackExtended` at `on_train_end` into a `hparams/` subdirectory. Visible in TensorBoard's **HParams** tab as a parallel coordinates / scatter plot for cross-run comparison. Hyperparameters tracked: `learning_rate`, `batch_size`, `num_epochs`, `spin_iteration`, `spin_lambda`.

| Tag | What it signifies | Trend |
|-----|-------------------|-------|
| `hparam/final_loss` | Final loss for this hyperparameter combination | ↓ decreases |
| `hparam/final_margin` | Final SPIN margin | ↑→ rises |
| `hparam/final_win_rate` | Final win rate | ↑→ rises |
| `hparam/final_logp_gap` | Final log-prob gap | →↓ may narrow |
| `hparam/final_kl_from_ref` | Final signed avg policy advantage vs π_prev — positive = healthy alignment, negative = regression | → near 0; negative = check config |

---

#### `spin_progress/*` — per SPIN iteration (`global/`)

Logged by `SPINIterationSummaryCallback` at `on_train_end`, with **SPIN iteration index** as the x-axis. These are the primary signals for tracking SPIN convergence across all iterations on a single chart.

**Loss:**

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `spin_progress/final_loss` | Loss at the last step of the iteration | Last accumulated loss value | Loss value this iteration hands forward | ↓→ falls, plateaus near 0 |
| `spin_progress/mean_loss` | Average loss over all steps in the iteration | `mean(all step losses this iteration)` | More robust than final alone — a noisy last step can mislead | ↓→ falls, plateaus |
| `spin_progress/min_loss` | Best loss seen anywhere in the iteration | `min(all step losses)` | The lowest point reached; consistently falling confirms genuine improvement | ↓→ falls, plateaus |

**Alignment signal:**

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `spin_progress/final_margin` | SPIN margin at the last step | Last accumulated `margin_mean` | Higher = model strongly prefers human over synthetic at iteration end | ↑→ rises, plateaus |
| `spin_progress/mean_margin` | Average margin over all steps | `mean(margin_mean values this iteration)` | Iteration-level alignment health | ↑→ rises, plateaus |
| `spin_progress/final_pi_chosen_logp` | π_θ log-prob on human response at the last step | Last accumulated value | Becomes less negative each iteration as the model predicts human text better | ↑ increases (less negative) |
| `spin_progress/final_pi_rejected_logp` | π_θ log-prob on synthetic response at the last step | Last accumulated value | Also rises as synthetic quality improves; approaches `final_pi_chosen_logp` at Nash equilibrium | ↑ increases (less negative) |
| `spin_progress/final_logp_gap` | Log-prob gap at the last step | `final_pi_chosen_logp − final_pi_rejected_logp` | Should be positive and **growing** — the model is widening its preference for human over synthetic during active alignment. Only at Nash equilibrium (run near convergence) does the gap approach 0 as synthetic quality matches human quality | ↑ increases during active training; →0 only at Nash equilibrium |
| `spin_progress/mean_logp_gap` | Average logp gap over the iteration | `mean(pi_chosen_logp − pi_rejected_logp per step)` | More stable than the final value; follows the same widening trend | ↑ increases during active training; →0 only at Nash equilibrium |
| `spin_progress/final_win_rate` | Win rate at the last step | Last accumulated `win_rate` | How often the final model beats π_prev per example | ↑→ rises toward 1.0 |
| `spin_progress/mean_win_rate` | Average win rate over the iteration | `mean(win_rate values this iteration)` | Iteration-level discrimination ability | ↑→ rises toward 1.0 |
| `spin_progress/final_kl_from_ref` | Signed average policy advantage vs π_prev at the last step | `(mean(pi_chosen − ref_chosen) + mean(pi_rejected − ref_rejected)) / 2` | **Positive** = model improved on both sides. **Negative** = regressed on chosen or pushed rejected down harder than it lifted chosen. A cross-iteration downward trend means alignment is failing — reduce LR or λ | → near 0; negative trend = alignment alarm |
| `spin_progress/mean_kl_from_ref` | Average signed advantage over the iteration | `mean(kl_from_ref values this iteration)` | Per-iteration average. More robust than the final value — use alongside `final_kl_from_ref` | → near 0; negative trend = alignment alarm |

**Training configuration:**

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `spin_progress/total_steps` | Total optimizer steps in this iteration | `state.global_step` at `on_train_end` | Confirms the iteration ran for the expected number of steps | → stable |
| `spin_progress/final_lr` | Learning rate at the **very last logged step** of the iteration | Last `learning_rate` value accumulated from logs | HF Trainer's default linear decay schedule reduces the LR from the configured peak (e.g. 5e-5) to ~0 by the final step of each training call. So `final_lr` is expected to be near zero every iteration — it does **not** represent the peak or average LR. To see the peak LR configured for each iteration use `train/learning_rate` early in the training steps | → near 0 every iteration (linear decay bottoms out); configured peak is the meaningful number |
| `spin_progress/spin_lambda` | λ value used for this iteration | `lambda_initial` or `lambda_final_iteration` | Confirms λ jumped on the final iteration as configured | →↑ stable, then jumps on the last iteration |
| `spin_progress/dataset_size` | Number of training examples this iteration | Set by `set_iteration()` before `trainer.train()` | Useful when dataset size varies across iterations | → stable |

**Weight drift:**

| Tag | What it is | How it's calculated | What it signifies | Trend |
|-----|-----------|---------------------|-------------------|-------|
| `spin_progress/weight_drift_from_iter_start` | How much the model changed within this iteration | `sqrt(Σ ‖w_final − w_iter_start‖²)` across all trainable params | Magnitude of the alignment update this round. Sudden large jumps with poor metrics indicate instability | ↓→ may decrease slightly as model nears convergence |
| `spin_progress/weight_drift_from_base_model` | Cumulative drift from the very first checkpoint | `sqrt(Σ ‖w_final − w_base‖²)` — base snapshot captured at iteration 0 | Total alignment shift across all iterations combined | ↑ always increases |
| `spin_progress/cosine_sim_to_iter_start` | Directional similarity between iteration-start and iteration-end weights | `cosine_similarity(flat_start_params, flat_end_params)` over all `requires_grad=True` parameters | With LoRA, `lora_B` is **reinitialized to 0** at the start of every batch via `make_trainable()`. As training becomes more effective, `lora_B` accumulates larger values by the end of each iteration — the weight vector drifts further from the all-zero starting point, producing **lower** cosine similarity. Near convergence the updates shrink and similarity may recover slightly, but the dominant trend during active alignment is downward. Values well below 0.5 with large λ are normal | ↓ decreases as training becomes more effective; may stabilise near convergence |

---

#### Non-scalar outputs

| Panel / Tab | Tag or location | What it shows | Trend across iterations |
|------------|-----------------|---------------|------------------------|
| **Text → config/spin_config** | `iter_N/batch_K/` at training start | Full `SPINConfig` dump — confirms active hyperparameters | → same every iteration |
| **Text → spin_iterations/log** | `global/` at each iteration start | One-line record of LR, epochs, batch size for each SPIN iteration | LR entry ↓ after `late_lr_start_iteration` |
| **Text → spin_iterations/summary** | `global/` at each iteration end | Markdown card: final loss, margin, win rate, logp gap, KL, λ, steps, dataset size | Values follow their respective scalar trends |
| **Text → profiler/key_averages_iter_N** | `global/` if `enable_profiler=True` | Top-N PyTorch operators sorted by CUDA time — operator-level GPU profile | → similar op profile each iteration |
| **PR Curves → train/alignment_pr_curve** | `iter_N/batch_K/` per epoch if `log_pr_curves=True` | Label = 1 when margin > 0; score = sigmoid(margin). Precision-Recall curve showing how reliably margin predicts alignment | ↑ AUC rises toward 1.0 across iterations |
| **Projector → embeddings/tokens_iter_start** | `iter_N/batch_K/` at training start if `log_embedding_projector=True` | PCA/UMAP/t-SNE of the token embedding matrix before training | Visible cluster shift across iterations as alignment reshapes token geometry |
| **Projector → embeddings/tokens_iter_end** | `iter_N/batch_K/` at training end | Same embedding matrix after training — compare to iter_start | ↓ smaller shift from start→end as model converges |
| **Histograms → epoch/loss_distribution** | `iter_N/batch_K/` per epoch | Distribution of step-level loss values in the epoch | ↓ distribution shifts toward 0 across iterations |
| **Graphs** | `global/` if `log_model_graph=True` | Schematic causal LM architecture (embedding → N decoder layers → LM head) | → same every iteration |

---

## Project Structure

| File | Description |
|------|-------------|
| [main.py](main.py) | Entry point — outer SPIN loop, resume logic, orchestration |
| [evaluate.py](evaluate.py) | Benchmark evaluation — `run_eval(cfg)` drives five tasks, iteration-over-iteration comparison, and TensorBoard logging; delegates model loading, memory management, and I/O to utils.py |
| [spin_config.py](spin_config.py) | `SPINConfig` dataclass — all training and evaluation hyperparameters with inline docs |
| [spin_trainer.py](spin_trainer.py) | `SPINTrainer` and `RMSPropSPINTrainer` — loss computation and training step |
| [spin_dataset.py](spin_dataset.py) | `SPINDataset` — pre-tokenises chosen/rejected pairs |
| [spin_data_collator.py](spin_data_collator.py) | `SPINDataCollator` — pads and batches chosen/rejected tensors |
| [utils.py](utils.py) | Shared infrastructure for training and evaluation: model loading, tokenisation, generation, LoRA merge, memory logging, directory/file helpers, arg parsing |
| [trainer_callback/](trainer_callback/) | TensorBoard, profiler, memory probe, and iteration summary callbacks |


---

## Implementation Deep-Dive

### Outer Loop — [main.py](main.py)

`main.py` is the entry point and implements the outer SPIN loop. Its responsibilities:

1. **Parse config** — argparse flags are dynamically generated from the `SPINConfig`
   dataclass fields so every hyperparameter is exposed as a CLI flag without maintaining
   a separate argparse setup.
2. **Setup** — Creates output directories and saves a `config.json` snapshot.
3. **Load tokenizer and base dataset** — The dataset is fully materialised into a Python
   list of `{prompt, response}` dicts in **fixed index order** (no shuffling) and reused
   across all iterations.
4. **Detect resume point** — `find_start_iteration()` scans `.done` sentinels to skip
   already-complete iterations; `_find_start_batch()` scans per-batch sentinels to skip
   already-complete batches within the current iteration.
5. **Run the iteration loop** — For each iteration: loads `π_prev` from disk to CPU as a
   frozen reference model, then runs the inner batch loop (Steps 1–3 per batch).
6. **Evaluate the new checkpoint** — When `eval_run_after_training` is `True` (default),
   calls `evaluate.run_eval(cfg)` immediately after that iteration's checkpoint is saved.
   `run_eval()` auto-discovers every `iter_*` directory and skips any with a cached
   `.parsed.json`, so only the iteration that just finished is actually benchmarked.
7. **Global TensorBoard** — A `SummaryWriter` at `tensorboard/global/` captures
   cross-iteration memory and training signals.

**Key functions:**

| Function | Role |
|---|---|
| `_find_start_batch(cfg, iteration, total_batches)` | Scans batch files in order; returns first k where synth, logprobs, or `.done` is missing |
| `_init_train_model(prev_model, cfg, iteration, start_batch)` | Returns trainable model: wraps prev_model for batch 0, or loads merged checkpoint from batch k−1 for resume |
| `_step_synth(prev_model, tokenizer, chunk, cfg, iteration, k)` | Step 1 — generates and saves synth; skips if file exists |
| `_step_logprobs(prev_model, tokenizer, synth_rows, cfg, iteration, k)` | Step 2 — scores and saves logprobs; skips if file exists |
| `_step_train(train_model, synth_rows, ref_lps, tokenizer, cfg, ...)` | Step 3 — trains, merges LoRA, saves model, writes `.done`; skips if `.done` exists |
| `_cleanup_trainer_checkpoints(directory)` | Deletes HF Trainer `checkpoint-N` subdirs after `save_model()` completes; the merged model supersedes them |

**Memory management within an iteration:**
- `π_prev` lives on CPU throughout the iteration, moving to GPU only for Steps 1–2
  of each batch, then back to CPU. This frees the GPU for the training step.
- `train_model` stays in GPU memory between batches — no disk reload between batches
  in the normal (non-resume) case.
- After each batch's training, LoRA is merged into base weights, and `make_trainable()`
  re-applies fresh adapters for the next batch.

### Configuration — [spin_config.py](spin_config.py)

All hyperparameters live in the `SPINConfig` dataclass. Every field carries inline
documentation describing its effect and recommended range. Key sections:

**Model and dtype:**
- `model_name_or_path` — HuggingFace Hub ID or local path.
- `torch_dtype` — `"bfloat16"` (default, recommended on Ampere+ GPUs), `"float16"`,
  or `"float32"`.
- `attn_implementation` — `"sdpa"` (default, free on PyTorch 2.0+) or
  `"flash_attention_2"` (requires the `flash-attn` package, 2–4× faster).

**SPIN-specific:**
- `lambda_initial` (default 0.1) — scales the margin for all iterations except the last.
- `lambda_final_iteration` (default 5.0) — larger λ for the final iteration applies a
  stronger alignment push.
- `loss_type` — `"logistic"` (smooth, never saturates, recommended), `"hinge"`,
  `"correlation"`, or `"exponential"`.

**LoRA:**
- `use_lora=True`, `lora_r=16`, `lora_alpha=32` — default `lora_target_modules` is
  `c_attn,c_proj` (GPT-2/TinyStories style). For LLaMA/Mistral/Phi/SmolLM models use
  `q_proj,k_proj,v_proj,o_proj`. `make_trainable()` auto-detects the correct target
  layers for all supported architectures when the configured names are not found in
  the model — no manual override needed when switching between model families.

**Two-phase learning rate:**
- `learning_rate=5e-5` for early iterations, `learning_rate_late=1e-5` from
  `late_lr_start_iteration` onward. This allows a gentle step-down LR schedule across
  iterations without a per-step scheduler.

**torch.compile:**
- `compile_model=True` with `compile_backend="inductor"` and
  `compile_mode="max-autotune-no-cudagraphs"` (the `no-cudagraphs` variant avoids a
  C++ OpenMP dependency on Windows while still enabling Triton kernel tuning).
- `compile_ref_model=False` — the reference model is not compiled because
  `model.generate()` uses a Python while-loop that causes graph breaks, so
  `compile_fullgraph=True` would silently fall back to eager mode anyway.

### Dataset — [spin_dataset.py](spin_dataset.py)

`SPINDataset` is a standard PyTorch `Dataset` that pre-tokenises all training rows at
construction time. For each row it tokenises both the chosen (human) and rejected
(synthetic) sequences and stores the results.

Tokenisation performs several careful steps:

1. **Format the prompt** — applies the chat template (tokenizer's built-in template,
   `instruction_response` manual wrapping, or plain passthrough) depending on
   `chat_template_mode`.
2. **Tokenize together** — the prompt and response are concatenated *before*
   tokenisation to avoid "boundary artifacts" — tokenisers can split subwords
   differently when strings are encoded in isolation vs concatenated.
3. **Mask prompt tokens** — labels for prompt token positions are set to `-100` so the
   loss is computed only over the response tokens, not the conditioning context.

```python
# Resulting dict for one sequence:
{
    "input_ids":       [101, 234, 567, ...],   # full prompt+response token ids
    "attention_mask":  [1, 1, 1, ...],
    "labels":          [-100, -100, 567, ...]  # -100 masks prompt positions from loss
}
```

The dataset also stores a `length` key (the maximum of chosen/rejected lengths) used by
HuggingFace's `LengthGroupedSampler` to sort batches by length, minimising padding
overhead. If reference log-probs are provided (always during training), each item also
carries `ref_chosen_logp` and `ref_rejected_logp` as scalars.

### Data Collator — [spin_data_collator.py](spin_data_collator.py)

`SPINDataCollator` receives a list of dataset items (one per example in the batch) and
pads them into batched tensors:

- `chosen_input_ids` / `rejected_input_ids` — right-padded to the longest sequence
  in the batch with `pad_token_id`.
- `chosen_attention_mask` / `rejected_attention_mask` — right-padded with `0`.
- `chosen_labels` / `rejected_labels` — right-padded with `-100` so padded positions
  are automatically excluded from the loss.
- `ref_chosen_logp` / `ref_rejected_logp` — stacked into a `(batch,)` float32 tensor.

The chosen and rejected sequences are padded *independently* (they may have very
different lengths). This is critical for the memory-efficient two-pass forward during
training — each pass only needs to allocate memory for its own sequence length.

### Trainer and Loss — [spin_trainer.py](spin_trainer.py)

`SPINTrainer` extends HuggingFace's `Trainer` and overrides the training step.

#### Training Step

This is the hot path — called once per micro-batch:

1. The reference log-probs (`ref_chosen_logp`, `ref_rejected_logp`) are loaded from the
   batch as pre-computed scalars — no model forward pass is needed.
2. **Forward pass 1** — runs the trainable model on the chosen (human) sequence to get
   `log π_θ(y_human | x)`.
3. **Forward pass 2** — runs the trainable model on the rejected (synthetic) sequence to
   get `log π_θ(y_synthetic | x)`.
4. **Margin** — computed as `(π_θ(chosen) − ref_chosen_logp) − (π_θ(rejected) − ref_rejected_logp)`,
   then scaled by λ.
5. **Loss** — the configured loss function is applied to the margin and backpropagation runs.

**Why two separate forward passes instead of one?**
Batching chosen and rejected together would require a batch of size `2 × batch_size`
with both sequence types, forcing peak activation memory to equal `chosen_len +
rejected_len`. Running them sequentially caps peak memory at `max(chosen_len,
rejected_len)` — up to 2× less for long sequences.

Per-step metrics logged: `loss`, `margin_mean`, `margin_std`, `win_rate`
(fraction of examples where `margin > 0`), `pi_chosen_logp`, `pi_rejected_logp`,
`ref_chosen_logp`, `ref_rejected_logp`, `kl_from_ref`, `learning_rate`.

#### RMSProp Variant

`RMSPropSPINTrainer` subclasses `SPINTrainer` and overrides the optimizer creation to
use RMSprop instead of AdamW. RMSprop maintains one optimizer-state tensor per
parameter (running mean of squared gradients) versus AdamW's two (first + second
moment), saving approximately 2 GB of GPU memory per 1B-parameter model. The
`foreach=True` flag enables fused kernel implementations for the update step,
recovering some of the speed cost.

#### SPIN Loss

The loss operates on a per-example `margin` (after λ scaling):

| `loss_type` | Formula | Gradient behaviour |
|---|---|---|
| `logistic` | `softplus(−margin)` = `log(1 + exp(−margin))` | Smooth; always non-zero gradient; asymptotes to 0 from above as margin→∞ |
| `hinge` | `relu(1 − margin)` | Zero gradient when `margin > 1`; hard boundary |
| `correlation` | `1 − margin` | Constant gradient regardless of margin |
| `exponential` | `exp(−margin)` | Very aggressive gradient for negative margins; can cause instability |

`logistic` is the default and recommended setting — it never fully stops penalising
negative margins, which keeps gradients flowing even on nearly-aligned examples.

### Utilities — [utils.py](utils.py)

`utils.py` provides all shared infrastructure used by both `main.py` (training) and
`evaluate.py` (evaluation). `evaluate.py` imports everything via `from utils import *`
and delegates model loading, tokenisation, compilation, directory management, memory
logging, and JSON I/O to the helpers defined here.

**Model lifecycle:**
- Loading `AutoModelForCausalLM` with the configured dtype and attention implementation.
  When loading as the frozen reference model, `eval()` mode is set and all parameter
  gradients are disabled. When loading as the trainable model, gradient checkpointing
  is optionally enabled.
- Converting from frozen reference to trainable: base weights are frozen and PEFT LoRA
  adapters are inserted via `get_peft_model()`, or all parameters are unfrozen for full
  fine-tuning.
- After training, LoRA adapters are merged back into the base weights via
  `merge_and_unload()` before saving.
- Between iterations the model is deleted, garbage collection runs, and the CUDA cache
  is emptied to free GPU memory.

**Log-probability computation:**
- A forward pass through the model produces logits of shape `(batch, seq_len, vocab)`.
- The standard autoregressive shift is applied: logits at position `t` predict token
  `t+1`.
- Prompt positions (where `labels == -100`) are masked out.
- Per-token log-probabilities are gathered and summed over response tokens to produce
  one scalar per sequence.
- `use_cache=False` prevents unnecessary KV-cache allocation during scoring.

**Generation:**
- Prompts are batched and fed to `model.generate()`. The attention mask's row sums give
  the actual prompt lengths (handling left-padded batches), which are used to slice the
  newly generated tokens from the output sequences.

**Tokenisation:**
- Three chat-template modes: `"plain"` (raw prompt), `"instruction_response"` (manual
  prefix/suffix wrapping), and `"auto"` (uses the tokenizer's built-in `chat_template`
  if available, falls back to `instruction_response`).
- The prompt and response are always concatenated before tokenisation to avoid
  subword-boundary artifacts.

**Dataset loading:**
- Supports HuggingFace Hub datasets and local JSONL/JSON/Parquet files.
- Multi-turn chat datasets are normalised by extracting the first user/assistant turn,
  tolerating role-name variants (`user`, `human`, `assistant`, `model`, `gpt`, `bot`).
- Records missing a valid user→assistant exchange are silently skipped.

**Crash safety:**
- All JSONL caches are written via a write-to-temp-then-rename strategy so a kill
  mid-write never leaves a corrupt file.
- Stale HuggingFace `.lock` files from killed prior runs are cleaned up at process
  start to prevent deadlocks.

**Training arg helpers:**
- λ selection: returns `lambda_final_iteration` on the last iteration and
  `lambda_initial` for all others.
- LR selection: `learning_rate` for early iterations, `learning_rate_late` from
  `late_lr_start_iteration` onward.

### Callbacks — [trainer_callback/](trainer_callback/)

Four callbacks augment training with observability:

| Callback | Purpose |
|---|---|
| `TensorBoardCallbackExtended` | Per-step scalars (loss, margin, win rate, log-probs, KL, LR), PR curves, embedding projector snapshots, model graph |
| `TensorBoardParameterStatsCallback` | Per-parameter weight/gradient histograms and scalar statistics (mean, std, L2 norm) every `parameter_log_interval` steps |
| `TorchProfilerCallback` | PyTorch profiler traces: operator-level GPU/CPU timelines, memory events, FLOP counts, flamegraph stacks |
| `MemoryProbeCallback` | CPU RSS and GPU allocated/reserved memory at the start/end of each epoch |
| `SPINIterationSummaryCallback` | Cross-iteration summary written to the global TensorBoard run; tracks dataset size, lambda, win rate trends across iterations |

---

## Evaluation — [evaluate.py](evaluate.py)

`evaluate.py` is a benchmark harness that evaluates every trained checkpoint against
five standard LLM benchmarks (GSM8k is included in the implementation but disabled by
default — uncomment its line in `TASKS` to enable it). It requires no `lm_eval`
dependency — all scoring is implemented directly using HuggingFace `transformers`.

Model loading, tokeniser setup, compilation, memory management, and JSON output all
delegate to the shared helpers in `utils.py` (imported via `from utils import *`):

| utils.py function | Role in evaluate.py |
|---|---|
| `load_causal_lm(path, cfg, trainable=False)` | Load each checkpoint in frozen eval mode |
| `load_tokenizer(cfg)` | Configure tokeniser (pad token, padding side) from checkpoint path |
| `maybe_compile_model(model, cfg, label)` | `torch.compile()` the eval model with `fullgraph=False` |
| `ensure_dir(path)` | Create output and TensorBoard directories |
| `free_model(model)` | Delete model, run GC, empty CUDA cache between checkpoints |
| `save_json(path, obj)` | Write per-iteration score files and the final summary |
| `log_memory(tag)` | Log CPU/GPU memory before and after each model load/free |

### Programmatic vs Standalone Usage

**Automatic, per-iteration** (default): `main.py` calls `evaluate.run_eval(cfg)` right
after each SPIN iteration's checkpoint is saved, whenever `cfg.eval_run_after_training`
is `True` (the default). Because `run_eval()` auto-discovers every `iter_*` directory
and skips any with a cached `.parsed.json`, each call only benchmarks the iteration
that just finished — earlier iterations are loaded from cache, not re-evaluated. Set
`eval_run_after_training=False` to disable this and only evaluate manually.

**Standalone** (CLI):
```bash
python evaluate.py --checkpoints-dir ./spin_outputs/checkpoints
```
`main()` parses CLI flags (defaults derived from `SPINConfig()`), builds a config
with `dataclasses.replace()`, and calls `run_eval(cfg)`.

**Programmatic** (called from another script):
```python
from evaluate import run_eval
run_eval(cfg)                                  # all iterations, all tasks
run_eval(cfg, iters=["iter_2", "iter_4"])      # specific iterations
run_eval(cfg, active_tasks=[...], n_shots={…}) # custom task subset / shot counts
```

All evaluation settings (`eval_output_dir`, `eval_batch_size`, `eval_limit`, etc.)
come from `SPINConfig` fields — see [Configuration Reference](#configuration-reference).

### Overview Flow

```
run_eval(cfg)
├── ensure_dir(cfg.eval_output_dir)
├── ensure_dir(cfg.eval_tensorboard_dir)
├── Discover iter_* checkpoint directories (sorted by numeric index)
└── For each iteration:
    ├── [cache hit] load scores from iter_N.parsed.json
    └── [cache miss]
        ├── find_model_path() — probe candidate sub-dirs for config.json
        ├── load_tokenizer(cfg with tokenizer_name_or_path=checkpoint_path)
        ├── log_memory(before_load)
        ├── load_causal_lm(path, cfg, trainable=False).to(cfg.device)
        ├── log_memory(after_load)
        ├── maybe_compile_model() with fullgraph=False, mode="default" (safe for model.generate)
        ├── run_all_benchmarks() → per-task scores
        ├── log_memory(before_free) → free_model() → log_memory(after_free)
        └── save_json(iter_N.parsed.json, scores)
    ├── Compute delta vs previous iteration
    ├── Track running best-average
    └── Write TensorBoard row
├── Print formatted comparison table to stdout
├── write_summary() → comparative_summary.txt
├── save_json() → comparative_summary.json
└── Close TensorBoard writer
```

### Benchmark Descriptions

| Benchmark | Metric | Shots | What it tests | Active |
|---|---|---|---|---|
| **ARC-Challenge** | acc_norm | 25 | Grade-school science questions selected to defeat retrieval and word-co-occurrence methods | ✅ |
| **TruthfulQA MC2** | mc2 | 0 | Whether the model outputs truthful statements; multiple correct answers per question | ✅ |
| **Winogrande** | acc | 5 | Commonsense pronoun/coreference resolution (large-scale Winograd schema) | ✅ |
| **HellaSwag** | acc_norm | 10 | Commonsense sentence completion; adversarially selected incorrect endings | ✅ |
| **MMLU** | acc | 5 | 57 academic subjects spanning humanities, STEM, social sciences, and professional domains | ✅ |
| **GSM8k** | acc | 5 | Grade-school arithmetic word problems requiring multi-step chain-of-thought reasoning | ⬜ disabled by default |

GSM8k is implemented but commented out in the `TASKS` list — uncomment it to enable.
Shot counts match the Open LLM Leaderboard v1 defaults so results are directly
comparable to published numbers.

### Scoring Strategy

All multiple-choice benchmarks (ARC, TruthfulQA, Winogrande, HellaSwag, MMLU) use
**log-likelihood continuation scoring**. For each answer choice, the model scores:

```
sum_t log P(choice_token_t | context, choice_tokens_0..t-1)
```

The choice with the highest score wins.

**Key implementation detail — tokenise together, not separately:**
The context and each choice are always concatenated *before* tokenisation. Tokenisers
can split subwords differently at a string boundary when the two strings are encoded in
isolation (the "boundary artifact" problem). Tokenising the full string together ensures
the model sees exactly the tokens it would during free-form generation.

**Truncation:** When the combined sequence exceeds `MAX_SEQ_LEN=2048` tokens, it is
truncated from the *left* of the context window, preserving the choice tokens intact —
since those are the tokens being scored.

**Batched scoring:** `score_examples_batched()` amortises GPU kernel-launch overhead by
packing `eval_batch_size` (context, continuation) sequences from *multiple questions* into
a single forward pass — choices from different questions are batched together, not just
choices from the same question. Sequences are left-padded to the batch maximum length so
real tokens are right-aligned and attention patterns are valid. This gives higher GPU
utilisation than one pass per question or per choice.

**acc_norm (length normalisation):** Used for ARC-Challenge and HellaSwag. Before
selecting the winning choice, each score is divided by the *character length* of the
choice text. Without normalisation the model would trivially prefer shorter choices
that accumulate fewer (negative) log-probabilities.

**mc2 (TruthfulQA):** TruthfulQA MC2 has *multiple* correct answers per question.
Softmax is applied across all choice log-likelihoods to produce a probability
distribution; the score is the sum of probability mass landing on all correct choices.
A score of 1.0 means all probability was assigned to true statements.

**GSM8k uses greedy generation** instead of log-likelihood scoring. The model freely
generates its answer and the final number is extracted — first by looking for the
canonical `"#### N"` delimiter used in the training data, falling back to the last
number anywhere in the generated text. Comma stripping makes `"1,234"` and `"1234"`
compare equal.

### Per-Task Details

#### ARC-Challenge

- **Dataset:** `allenai/ai2_arc` (ARC-Challenge split), test set.
- **Few-shot format:** `"Question: {question}\nAnswer: {full answer text}"` × 25
  exemplars from the training split. The full answer text (not just the letter A/B/C/D)
  is used both in exemplars and as the scored continuation.
- **Scoring:** acc_norm — length-normalised log-likelihood over all choice texts.

#### TruthfulQA MC2

- **Dataset:** `truthful_qa` (multiple_choice split), validation set.
- **Format:** `"Q: {question}\nA: {choice}"` (zero-shot only — no reliable few-shot
  training split exists).
- **Scoring:** mc2 — softmax over all choice log-likelihoods; sum probability mass on
  the subset of correct choices.

#### Winogrande

- **Dataset:** `winogrande` (winogrande_xl split), validation set.
- **Format:** The sentence has a blank `_`. Context = text before the blank. Each option
  is scored as `"{option}{text after blank}"`.
- **Example:** *"The trophy doesn't fit in the suitcase because _ is too large."*
  → Context: *"The trophy doesn't fit in the suitcase because "*, scored continuations:
  *"the trophy is too large."* vs *"the suitcase is too large."*
- **Scoring:** acc — raw log-likelihood, no normalisation.

#### GSM8k

- **Dataset:** `gsm8k` (main split), test set.
- **Few-shot:** 5 hard-coded Chain-of-Thought exemplars teach the model to show its
  work and end with `"#### <number>"`. These are the canonical exemplars from Wei et
  al. (2022) and lm_eval.
- **Scoring:** acc — greedy generation followed by exact numeric string match after
  extracting the final answer from both the generated text and the ground truth.

#### HellaSwag

- **Dataset:** `Rowan/hellaswag`, validation set.
- **Preprocessing:** `[bracket]` annotation artefacts and extra whitespace are stripped.
- **Format:** `"{activity label}: {partial context}"` + `" {ending}"`.
- **Scoring:** acc_norm — length-normalised log-likelihood over four candidate endings.

#### MMLU

- **Dataset:** `cais/mmlu` (all subjects), test set.
- **Format:**
  ```
  The following is a multiple choice question about {subject}.
  {question}
  A. {choice0}  B. {choice1}  C. {choice2}  D. {choice3}
  Answer:
  ```
  The continuation is a single letter ` A`, ` B`, ` C`, or ` D`.
- **Few-shot:** Per-subject few-shot prefix using up to 5 examples from the `dev` split
  for the same subject as the test question.
- **Scoring:** acc — no normalisation (all choices are the same length: one character).

### Evaluation Loop and Output

**Model discovery:** The evaluator probes a prioritised list of candidate
sub-directories inside each `iter_*` folder (`hf_final`, `final_checkpoint`,
`checkpoint-final`, `merged`, `model`, the folder itself) looking for `config.json` or
`adapter_config.json`. If none match it walks the entire subtree recursively. This
handles all checkpoint layouts SPIN may produce.

**Memory management:** `free_model()` (from utils.py) deletes the model reference, runs
garbage collection, and empties the CUDA cache immediately after benchmarks complete.
`log_memory()` records CPU RSS and GPU allocated/reserved memory before load and after
free, so memory growth across iterations is visible in the log. Loading and scoring a
single checkpoint can require 4–16 GB depending on model size; freeing between
iterations prevents OOM when evaluating many checkpoints in sequence.

**Delta tracking:** Per-task score differences between consecutive iterations are
computed. `None` is returned for any task that failed in either iteration so that
`0.0` unambiguously means no change (not a missing result).

**Best-iteration tracking:** The running best average across all evaluated iterations
is tracked and annotated with `★ NEW BEST` in both the CLI output and TensorBoard.

**Output files:**
```
eval_results/
├── iter_0.parsed.json          # Per-task scores for iter_0 (raw fractions × 100)
├── iter_1.parsed.json
├── comparative_summary.txt     # TSV table + per-iteration narrative (paste into spreadsheet)
└── comparative_summary.json    # Full results list with deltas and best-so-far tracking
```

**CLI output example:**
```
============================================================
 iter_1  →  /path/to/checkpoints/iter_1
============================================================
  [Arc] arc_challenge | 25-shot | full ...
    Arc: 42.15%  (183s)
  [TruthfulQA] truthfulqa_mc2 | 0-shot | full ...
    TruthfulQA: 51.30%  (97s)
  ...

  [iter_1]  avg=46.72%  Δprev_avg=▲+1.40  best_so_far=46.72%  ★ NEW BEST  (712s total)
  ▲ improved: Arc, TruthfulQA, HellaSwag
  ▼ declined: Winogrande
```

**Final comparison table:**
```
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
| Iteration | Arc  | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU  | Avg%  | ΔAvg  | Status |
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
| iter_0   | 40.80 | 50.10      | 62.30     | 18.50 | 71.20     | 46.00 | 48.15 |   NA  |        |
| iter_1   | 42.15 | 51.30      | 61.90     | 19.20 | 72.40     | 47.30 | 49.04 | ▲+0.89| ★ BEST |
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
```

### Evaluation TensorBoard Panels

```bash
tensorboard --logdir ./spin_outputs/tensorboard
```

The evaluation run writes to `tensorboard/eval_compare/`:

| Panel | Tags | Description |
|---|---|---|
| **Custom Scalars → Evaluation** | `eval/average`, `eval/best_so_far_average` | Average score and running best across iterations |
| **Custom Scalars → Evaluation** | `eval/tasks/<name>` | Per-task score for each iteration |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/<name>` | Per-task score change from the preceding iteration |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/improvement_rate` | Fraction of tasks that improved (0–1) |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/improved_task_count`, `…/declined_task_count` | Count of tasks improved / declined |
| **Text → eval/scorecard** | — | Markdown table of scores + deltas, one card per iteration |
| **Text → eval/best_iteration** | — | Note written each time a new best average is reached |
| **Text → eval/run_config** | — | Shot counts, device, directories — written once at step 0 |

---




---





## Detailed Explanation

### The Game-Theoretic View

SPIN frames alignment as a **two-player zero-sum game**:

- **The opponent** (`π_prev`, the *frozen* checkpoint from the previous iteration)
  generates synthetic responses for every training prompt. Its "move" is to produce
  completions that are as close to human quality as it can manage — making it hard for
  `π_θ` to tell the synthetic from the real.
- **The main player** (`π_θ`, the *trainable* model) does not debate `π_prev` in any
  literal sense. Instead, gradient descent updates `π_θ` to assign *higher*
  log-probability to the human response than to `π_prev`'s synthetic response — for
  the same prompt. Each gradient step is a "move" by the main player. It "wins" a
  training example when the human response scores higher than the synthetic by a
  positive margin (see the margin formula below).

Each iteration the winning `π_θ` becomes the new frozen opponent, raising the bar for
the next round. This mirrors the self-play loops used in game-playing agents like
AlphaGo, applied here to language model alignment.

### The Mathematical Formulation

#### The Two Models in Play

At the start of each SPIN iteration two model instances exist:

```
  ┌───────────────────────────────────┐     ┌───────────────────────────────────┐
  │          π_prev  (FROZEN)         │     │          π_θ  (TRAINABLE)         │
  │                                   │     │                                   │
  │  Weights loaded from:             │     │  Starts as an exact copy of       │
  │    iter 0  →  base model          │     │  π_prev, then diverges as         │
  │    iter i  →  checkpoints/iter_   │     │  gradient descent updates it      │
  │              i-1/                 │     │                                   │
  │                                   │     │  LoRA adapters unlocked,          │
  │  eval() mode, no gradients        │     │  lives on GPU throughout          │
  │  Lives on CPU, moves to GPU       │     │  the training loop                │
  │  only for Steps 1 and 2           │     │                                   │
  └───────────────────────────────────┘     └───────────────────────────────────┘
           the "opponent"                            the "main player"
```

#### The Four Log-Probability Quantities

For every training example the margin formula needs **four scalars**. Each is the
sum of per-token log-probabilities that a model assigns to a full response given the
prompt — one large negative number per (model, response) pair.

```
Training example (one row from the dataset):

  x          =  "Explain the water cycle in simple terms."
                 ─────────────────────────────────────────
                 Comes from the training dataset. Same for both sides.

  y_human    =  "Water evaporates from oceans and lakes when heated by the sun..."
                 ──────────────────────────────────────────────────────────────────
                 Comes from the training dataset (the ground-truth human response).
                 This is the "chosen" / "good" side.

  y_synthetic =  "The water cycle refers to the continuous movement of water..."
                 ──────────────────────────────────────────────────────────────────
                 Generated by π_prev in Step 1 (synthetic generation).
                 This is the "rejected" / "bad" side.
```

| Symbol | Plain English | Source | Computed when |
|--------|--------------|--------|---------------|
| `log π_prev(y_human \| x)` | How likely is it that the **frozen** π_prev would produce the **human** response, token by token? | `ref_chosen_logp` field in `_logprobs.jsonl` | **Step 2** — scored once per batch; loaded from disk during training |
| `log π_prev(y_synthetic \| x)` | How likely is it that the **frozen** π_prev would produce the **synthetic** response it just wrote? | `ref_rejected_logp` field in `_logprobs.jsonl` | **Step 2** — scored once per batch; loaded from disk during training |
| `log π_θ(y_human \| x)` | How likely is it that the **trainable** π_θ would produce the **human** response right now? | Forward pass 1 inside `SPINTrainer.compute_loss()` | **Step 3** — recomputed every mini-batch; gradients flow through this |
| `log π_θ(y_synthetic \| x)` | How likely is it that the **trainable** π_θ would produce the **synthetic** response right now? | Forward pass 2 inside `SPINTrainer.compute_loss()` | **Step 3** — recomputed every mini-batch; gradients flow through this |


**`ref_chosen_logp` and `ref_rejected_logp`**
These two scalars are the **frozen baseline** — the log-probabilities `π_prev` assigns
to both responses *before any training happens in this iteration*. They answer: *"where
was the model before we updated it this round?"*

- `ref_chosen_logp = log π_prev(y_human | x)` — how confidently the frozen reference
  model predicts the human response, token by token. A model that has been aligned for
  several iterations assigns a less negative value here because it already expects
  high-quality responses. This is **not the training target** — it is the starting line.
  `π_θ` is rewarded only for going *further* than this baseline, not merely reaching it.

- `ref_rejected_logp = log π_prev(y_synthetic | x)` — how confidently the frozen model
  predicts its own generated output. Because `π_prev` produced `y_synthetic` by sampling
  from itself, it naturally assigns it relatively high probability. In early iterations,
  `ref_rejected_logp` can be close to (or even briefly exceed) `ref_chosen_logp` before
  alignment has taken hold.

Subtracting these baselines is what separates SPIN from SFT. Without the subtraction,
the loss would just push `π_θ` to assign maximum absolute probability to `y_human` in
isolation — equivalent to SFT, which can cause mode collapse. By measuring *relative
improvement* (how much `π_θ` moved *beyond* `π_prev`'s starting point), the margin
penalises a model that inflates probability on both responses equally, and rewards one
that selectively improves on the human response more than the synthetic one.

#### Data Flow — SPIN Iterations 0 and 1

The diagrams below trace every computed value across two full SPIN outer iterations —
where each number originates, what changes between iterations, and how the trained
model is handed forward to the next round. Optimizer-step detail is included within
each iteration to show how `π_θ`'s live log-probs shift during training while the
reference values from disk remain fixed throughout.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SPIN ITERATION 0   (no prior checkpoint — base model plays both roles initially)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  π_prev = base model (frozen, on CPU)
  π_θ    = exact copy of base model (LoRA adapters unlocked, on GPU)

  TRAINING DATASET (fixed — same rows used in every iteration and every batch):
    x       = "Explain the water cycle in simple terms."
    y_human = "Water evaporates from oceans and lakes when heated by the sun..."

 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 1 — Synthetic Generation   (π_prev = base model, moved to GPU)        │
 │                                                                              │
 │  For each prompt x in this data batch:                                       │
 │    x  →  π_base.generate(do_sample=True, temp=0.9, top_p=0.95)              │
 │        →  y_syn_0 = "The water cycle refers to the continuous movement..."   │
 │                      ─────────────────────────────────────────────────────   │
 │                      Weak output: base model is unaligned.                  │
 │                      Generic phrasing, may restate the question,            │
 │                      lacks the precision and structure of y_human.          │
 │                                                                              │
 │  Saved to disk (atomic write → rename):                                      │
 │    iter_0_batch_k_synth.jsonl                                               │
 │    {"prompt": x, "response": y_human, "synthetic_response": y_syn_0}        │
 │                                                                              │
 │  π_prev moved back to CPU.                                                  │
 └──────────────────────────────────────────────────────────────────────────┬───┘
                                                                            │
                                                                            ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 2 — Reference Scoring   (π_prev = base model, moved to GPU again)     │
 │                                                                              │
 │  Compute log-probability the base model assigns to each response:            │
 │                                                                              │
 │  ref_chosen_logp_0   = log π_base(y_human | x)    =  −2.42                 │
 │                         ─────────────────────────────────────────────────   │
 │                         Per-token average log-prob over y_human given x     │
 │                         (sum of per-token log-probs ÷ response token count).│
 │                         Less negative = higher P. Base model's view of the  │
 │                         human text. Typical range: −1 to −5 per token.      │
 │                                                                              │
 │  ref_rejected_logp_0 = log π_base(y_syn_0 | x)   =  −2.76                 │
 │                         ───────────────────────────────────────────────     │
 │                         Per-token average for the model's own generation.   │
 │                         More negative than ref_chosen_logp_0 → even the    │
 │                         unaligned base model ranks y_human higher per token.│
 │                         Chosen−rejected gap: −2.42 − (−2.76) = 0.34 nats.  │
 │                                                                              │
 │  Saved to disk:                                                              │
 │    iter_0_batch_k_logprobs.jsonl                                            │
 │    {"ref_chosen_logp": −2.42, "ref_rejected_logp": −2.76}                  │
 │                                                                              │
 │  π_prev moved back to CPU. These scalars are now the FIXED ANCHOR           │
 │  for all of Step 3 — they will not change until iteration 1.                │
 └──────────────────────────────────────────────────────────────────────────┬───┘
                                                                            │
                                                                            ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 3 — Training Loop   (π_θ on GPU; only 1 model on GPU at a time)       │
 │                                                                              │
 │  SPINTrainer.compute_loss() — called for every mini-batch:                  │
 │                                                                              │
 │  Loaded from disk (constant across ALL optimizer steps this iteration):     │
 │    ref_chosen_logp_0   = −2.42  ◄── iter_0_batch_k_logprobs.jsonl          │
 │    ref_rejected_logp_0 = −2.76  ◄── iter_0_batch_k_logprobs.jsonl          │
 │    (per-token averages; divide raw sum by response length)                  │
 │                                                                              │
 │  ─── Optimizer step 0  (π_θ weights are still an exact copy of π_prev) ─── │
 │                                                                              │
 │  Forward pass 1:  x ++ y_human →  π_θ [= base] → log_p = −2.42            │
 │  Forward pass 2:  x ++ y_syn_0 →  π_θ [= base] → log_p = −2.76            │
 │                                                                              │
 │  RI_human = log π_θ(y_human|x) − ref_chosen_logp_0                         │
 │           =       −2.42        −     (−2.42)         =   0.0                │
 │  RI_synth = log π_θ(y_syn_0|x) − ref_rejected_logp_0                       │
 │           =       −2.76        −     (−2.76)         =   0.0                │
 │                                                                              │
 │  margin = λ × (RI_human − RI_synth) = λ × (0.0 − 0.0) =   0.0             │
 │  loss   = softplus(−0.0) = log(2) ≈ 0.693                                   │
 │                                                                              │
 │  loss.backward() → optimizer.step()                                         │
 │  π_θ weights shift — it is no longer identical to the base model.           │
 │                                                                              │
 │  ─── Optimizer step 1  (π_θ weights have diverged from base model) ─────── │
 │                                                                              │
 │  Forward pass 1:  x ++ y_human →  π_θ [updated] → log_p = −2.22            │
 │                                                  ← shifted toward y_human   │
 │  Forward pass 2:  x ++ y_syn_0 →  π_θ [updated] → log_p = −2.66            │
 │                                                  ← mild incidental drift    │
 │                                                                              │
 │  RI_human = −2.22 − (−2.42) = +0.20   (π_θ now likes y_human 0.20 more)   │
 │  RI_synth = −2.66 − (−2.76) = +0.10   (π_θ drifted slightly on y_syn_0)   │
 │                                                                              │
 │  margin = λ × (0.20 − 0.10) = λ × 0.10  > 0  ✓  model is aligning         │
 │  loss   = softplus(−0.6λ)  < 0.693           (improving)                   │
 │                                                                              │
 │  loss.backward() → optimizer.step() → margin grows further with each step  │
 │                                                                              │
 │  … (repeats for num_epochs_per_iteration passes over this data batch) …     │
 │                                                                              │
 │  After all optimizer steps across all data batches:                         │
 │  LoRA adapters merged into base weights → saved as checkpoints/iter_0/     │
 └──────────────────────────────────────────────────────────────────────────┬───┘
                                                                            │
              ╔══════════════════════════════════════════════════════════╗  │
              ║  CHECKPOINT HANDOFF                                      ║◄─┘
              ║                                                          ║
              ║  checkpoints/iter_0/ = trained π_θ                      ║
              ║                                                          ║
              ║  Weights have shifted toward y_human and away from the  ║
              ║  weak synthetic outputs. This model is now better than  ║
              ║  the base model at producing human-quality responses.    ║
              ║                                                          ║
              ║  Next: iter_0 checkpoint becomes π_prev for iteration 1 ║
              ╚══════════════════════════════════════════════════════════╝
                                          │
                                          ▼

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SPIN ITERATION 1   (iter_0 checkpoint becomes the new frozen opponent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  π_prev = checkpoints/iter_0/ (frozen, on CPU)
  π_θ    = exact copy of iter_0 checkpoint (LoRA adapters unlocked, on GPU)

  TRAINING DATASET: same x and y_human as iteration 0.
  EVERYTHING ELSE (y_syn, ref log-probs, on-disk files) recomputed from scratch.

 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 1 — Synthetic Generation   (π_prev = iter_0 model, moved to GPU)      │
 │                                                                              │
 │  Same prompt x, but a STRONGER model generates the synthetic response:       │
 │    x  →  π_iter0.generate(do_sample=True, temp=0.9, top_p=0.95)             │
 │        →  y_syn_1 = "When sunlight heats water bodies, molecules gain        │
 │                      enough energy to escape as vapour, rising into..."      │
 │                      ─────────────────────────────────────────────────────   │
 │                      Stronger output: iter_0 model was already aligned once. │
 │                      Better vocabulary, factual precision, logical structure. │
 │                      y_syn_1 is harder to distinguish from y_human — the    │
 │                      opponent is now a tougher competitor.                   │
 │                                                                              │
 │  Saved to disk:                                                              │
 │    iter_1_batch_k_synth.jsonl                                               │
 │    {"prompt": x, "response": y_human, "synthetic_response": y_syn_1}        │
 │                                                                              │
 │  π_prev moved back to CPU.                                                  │
 └──────────────────────────────────────────────────────────────────────────┬───┘
                                                                            │
                                                                            ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 2 — Reference Scoring   (π_prev = iter_0 model, moved to GPU again)   │
 │                                                                              │
 │  ref_chosen_logp_1   = log π_iter0(y_human | x)    =  −2.12                │
 │                         ─────────────────────────────────────────────────   │
 │                         LESS negative than iter_0's −2.42 (per-token avg).  │
 │                         The iter_0 model assigns HIGHER probability to      │
 │                         y_human than the base model did — alignment worked. │
 │                         This value rises (becomes less negative) every      │
 │                         iteration as the model gets closer to human text.   │
 │                                                                              │
 │  ref_rejected_logp_1 = log π_iter0(y_syn_1 | x)   =  −2.40                │
 │                         ─────────────────────────────────────────────────   │
 │                         LESS negative than iter_0's −2.76 (per-token avg).  │
 │                         y_syn_1 is a stronger completion, so the iter_0    │
 │                         model considers it more probable than the weak      │
 │                         y_syn_0 was to the base model.                      │
 │                                                                              │
 │                         Chosen−rejected gap: −2.12 − (−2.40) = 0.28 nats   │
 │                         vs. iteration 0 gap:                   = 0.34 nats  │
 │                         ─────────────────────────────────────────────────   │
 │                         Gap narrows each iteration. The synthetic is harder │
 │                         to beat → training signal becomes subtler.          │
 │                                                                              │
 │  Saved to disk:                                                              │
 │    iter_1_batch_k_logprobs.jsonl                                            │
 │    {"ref_chosen_logp": −2.12, "ref_rejected_logp": −2.40}                  │
 │                                                                              │
 │  These scalars are the new FIXED ANCHOR for all of iter_1 Step 3.          │
 └──────────────────────────────────────────────────────────────────────────┬───┘
                                                                            │
                                                                            ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │  STEP 3 — Training Loop   (π_θ starts at iter_0 weights, not base weights)  │
 │                                                                              │
 │  Loaded from disk (constant throughout all iter_1 optimizer steps):         │
 │    ref_chosen_logp_1   = −2.12  ◄── iter_1_batch_k_logprobs.jsonl          │
 │    ref_rejected_logp_1 = −2.40  ◄── iter_1_batch_k_logprobs.jsonl          │
 │                                                                              │
 │  ─── Optimizer step 0  (π_θ weights are still identical to π_prev=iter_0) ─ │
 │                                                                              │
 │  Forward pass 1:  x ++ y_human →  π_θ [= iter_0] → log_p = −2.12          │
 │  Forward pass 2:  x ++ y_syn_1 →  π_θ [= iter_0] → log_p = −2.40          │
 │                                                                              │
 │  RI_human = −2.12 − (−2.12) = 0.0                                           │
 │  RI_synth = −2.40 − (−2.40) = 0.0                                           │
 │  margin = 0.0  →  loss = log(2) ≈ 0.693  (same starting loss as iter_0)    │
 │                                                                              │
 │  loss.backward() → optimizer.step()                                         │
 │  π_θ diverges from iter_0 π_prev.                                           │
 │                                                                              │
 │  ─── Optimizer step 1  (π_θ weights have moved) ─────────────────────────── │
 │                                                                              │
 │  Forward pass 1:  x ++ y_human →  π_θ [updated] → log_p = −2.02            │
 │  Forward pass 2:  x ++ y_syn_1 →  π_θ [updated] → log_p = −2.37            │
 │                                                                              │
 │  RI_human = −2.02 − (−2.12) = +0.10  (improvement on chosen)               │
 │  RI_synth = −2.37 − (−2.40) = +0.03  (smaller than iter_0's +0.10)         │
 │                          ─────────────────────────────────────────────────  │
 │                          y_syn_1 is tightly coupled to human quality →     │
 │                          π_θ drifts on it less easily. The model cannot     │
 │                          cheat by inflating both responses together.        │
 │                                                                              │
 │  margin = λ × (0.10 − 0.03) = λ × 0.07  > 0  ✓                            │
 │  loss   = softplus(−0.07λ)  < 0.693    (lower loss = better fit)            │
 │                                                                              │
 │  loss.backward() → optimizer.step() → margin continues to grow              │
 │                                                                              │
 │  After all optimizer steps across all data batches:                         │
 │  LoRA adapters merged → saved as checkpoints/iter_1/                       │
 └──────────────────────────────────────────────────────────────────────────────┘

  checkpoints/iter_1/ is now closer to the human data distribution than iter_0/.
  Next iteration: iter_1 becomes π_prev and the game gets harder still —
  converging toward the Nash equilibrium where π_θ ≈ p_data.
```

**Cross-iteration value summary:**

| Quantity | Iteration 0 | Iteration 1 | Trend |
|---|---|---|---|
| `π_prev` source | Base model | `checkpoints/iter_0/` | Gets stronger each round |
| `y_synthetic` quality | Weak — generic, imprecise | Stronger — resembles `y_human` | Harder to beat each round |
| `ref_chosen_logp` | −2.42 (base model, per-token avg) | −2.12 (iter_0's view) | Rises (less negative) each round |
| `ref_rejected_logp` | −2.76 | −2.40 | Rises (less negative) each round |
| Chosen − rejected gap | 0.34 nats/token | 0.28 nats/token | Narrows → subtler signal |
| `π_θ` starting weights | Base model | iter_0 checkpoint | Stronger starting point |
| `RI_synth` at opt step 1 | +0.10 | +0.03 | Harder to drift on synthetic |
| `margin` at opt step 1 | λ × 0.10 | λ × 0.07 | Margin from improved chosen, less drift on synthetic |

The `_logprobs.jsonl` values (`ref_chosen_logp`, `ref_rejected_logp`) are the **fixed
anchor** within each SPIN iteration — they never change across optimizer steps. They
change only at iteration boundaries, when a new, stronger `π_prev` recomputes them
from scratch against freshly generated synthetic responses.

#### The Formula — Step by Step

```
margin = λ × [ (log π_θ(y_human | x) − log π_prev(y_human | x))
              − (log π_θ(y_synthetic | x) − log π_prev(y_synthetic | x)) ]
```

Rename the two inner differences to make the structure visible:

```
  RI_human = log π_θ(y_human | x) − log π_prev(y_human | x)
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  "Relative Improvement on the human response"
  How many log-prob units has π_θ moved toward y_human compared to where π_prev was?
  Positive = π_θ now assigns MORE probability to y_human than π_prev did.
  Negative = π_θ now assigns LESS probability to y_human than π_prev did (regression).


  RI_synthetic = log π_θ(y_synthetic | x) − log π_prev(y_synthetic | x)
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  "Relative Improvement on the synthetic response"
  How many log-prob units has π_θ moved toward y_synthetic compared to where π_prev was?
  Positive = π_θ now assigns MORE probability to y_synthetic than π_prev did.


  margin = λ × (RI_human − RI_synthetic)
```

**The margin is positive when π_θ has moved further toward the human response
than toward the synthetic one — relative to where π_prev started.**

#### Worked Numerical Example

Suppose after one gradient step the log-probs look like this
(realistic values for a 50-token response — each is a large negative number
because it is a sum of negative per-token log-probs):

```
  ALIGNING CASE (margin > 0):

                        π_prev (frozen)    π_θ (current)    Δ = RI
                        ───────────────    ─────────────    ──────
  y_human  response:       −145.3            −142.8         +2.5   ← bigger
  y_synthetic response:    −148.7            −147.5         +1.2   ← smaller

  margin = λ × (RI_human − RI_synthetic)
         = λ × (  +2.5   −    +1.2   )
         = λ × 1.3   →  POSITIVE ✓
  π_θ moved more toward the good response. Loss is small; gradient is gentle.


  DIVERGING CASE (margin < 0):

                        π_prev (frozen)    π_θ (current)    Δ = RI
                        ───────────────    ─────────────    ──────
  y_human  response:       −145.3            −145.9         −0.6   ← regression
  y_synthetic response:    −148.7            −146.1         +2.6   ← large gain

  margin = λ × (RI_human − RI_synthetic)
         = λ × ( −0.6   −    +2.6   )
         = λ × −3.2   →  NEGATIVE ✗
  π_θ is chasing the synthetic response and forgetting the human one.
  Loss is large; gradient strongly corrects this.
```

#### Visualising the Margin on the Log-Prob Axis

```
  LOG-PROBABILITY AXIS  (more negative = model assigns LESS probability)

  ◄──────────────────────────────────────────────────────────────►
    less probable                                  more probable

  POSITIVE MARGIN — model is aligning:

  π_prev(y_synth)  π_θ(y_synth)    π_prev(y_human)     π_θ(y_human)
       │                │                 │                   │
  ─────●────────────────●─────────────────●───────────────────●─────
     −148.7           −147.5           −145.3              −142.8
            ├── RI_synth=1.2 ──┤         ├────── RI_human=2.5 ──────┤

  π_θ shifted the human response rightward by MORE than the synthetic → margin > 0


  NEGATIVE MARGIN — model is diverging:

  π_θ(y_human) π_prev(y_human)    π_prev(y_synth)              π_θ(y_synth)
       │               │                 │                           │
  ─────●───────────────●─────────────────●───────────────────────────●─────
     −145.9          −145.3           −148.7                      −146.1
     ├RI_human=−0.6┤                  ├──────── RI_synthetic=2.6 ─────────┤

  π_θ actually moved AWAY from the human response while chasing synthetic → margin < 0
```

#### Role of λ (Lambda)

λ is a scalar that is multiplied into the margin before the loss function sees it:

- **Small λ (e.g. 0.1, used for early iterations)** — even a large misalignment produces
  a modest gradient. Protects against overcorrection while the model is still far from
  the human distribution.
- **Large λ (e.g. 5.0, used for the final iteration only)** — amplifies the gradient signal
  for a strong final alignment push once the model is already close.

The two-phase schedule (`lambda_initial=0.1` early, `lambda_final_iteration=5.0` on the
last iteration) balances stability across early iterations with a decisive final correction.

#### Interpreting the Margin

| Sign of margin | What happened | Training effect |
|---|---|---|
| margin >> 0 | π_θ improved on human response much more than on synthetic | Loss ≈ 0; very small gradient — convergence |
| margin > 0 | π_θ improved more on human than synthetic | Loss decreasing; gradient reinforces direction |
| margin ≈ 0 | Improved equally on both (or not at all on either) | Weak gradient signal — plateau |
| margin < 0 | π_θ improved more on synthetic than human | Large loss; gradient corrects drift |
| margin << 0 | π_θ regressed on human while boosting synthetic | Very large gradient; may need lower LR |

The **loss** converts this margin to a scalar that is minimised by gradient descent.
Four loss types are available (see [Loss Functions](#spin-loss) below).

#### Margin Dynamics Over Time

The margin operates on three timescales that behave differently:

**Within a single SPIN iteration (optimizer steps 0 → N):**

```
  optimizer step 0  →  margin = 0.0    π_θ is still an exact copy of π_prev;
                                        RI_human and RI_synth are both 0.0 by definition.
  optimizer step 1  →  margin > 0      π_θ has started to prefer y_human over y_syn.
  optimizer step N  →  margin >> 0     π_θ strongly and consistently prefers y_human.
```

Margin should **increase** during training. A flat or negative margin after several steps
signals a problem: learning rate too low, λ too small, data batch too narrow, or the wrong
loss type for this stage.

**At iteration boundaries (iter i → iter i+1):**

Margin **resets to 0** at the start of every new iteration. This is correct and expected:
the new `π_prev` is the just-trained `π_θ`, so at step 0 they are identical — both RI
values are 0 and the margin is 0. The model quality has improved; the margin score has not,
because it measures improvement *relative to the current `π_prev`*, not in absolute terms.

**At Nash equilibrium (the convergence target):**

As the synthetic responses become indistinguishable from human responses, margin cannot
grow past ≈ 0 even with many optimizer steps. The loss gradient vanishes — the model has
matched the human data distribution and the training signal exhausts itself. This is the
desired endpoint, not a failure mode.

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  MARGIN OVER A FULL SPIN RUN  (schematic — not to scale)                    │
  │                                                                              │
  │  margin                                                                      │
  │    ▲                                                                         │
  │    │    iter 0              iter 1              iter 2  (convergence)        │
  │    │   ╱‾‾‾‾‾╲            ╱‾‾‾‾‾‾╲            ╱‾‾‾‾‾                        │
  │    │  ╱        ╲reset    ╱         ╲reset    ╱                               │
  │  0 ├─●──────────●───────●───────────●───────●──────────────────────────     │
  │    │  step 0 of  step 0  step 0 of   step 0  margin plateaus near 0;        │
  │    │  iter 0     of      iter 1       of      gradient vanishes              │
  │    │             iter 1               iter 2                                 │
  │                                                                              │
  │  ● = start of iteration (margin == 0; π_θ is a fresh copy of new π_prev)   │
  └──────────────────────────────────────────────────────────────────────────────┘
```

| Timescale | Expected behaviour | Warning sign |
|---|---|---|
| Within an iteration | Margin increases from 0 toward a positive plateau | Flat or negative after many steps |
| Iteration boundary | Margin resets to 0 — always | Any non-zero starting margin indicates a bug |
| Across iterations | Per-iteration final margin stays similar or improves; `win_rate` trends up | Falling `win_rate` across iterations |
| At convergence | Margin cannot grow above ≈ 0 despite training | None — stop here |

### Why Does This Work?

The key property that makes SPIN converge is:

1. At iteration 0, `π_prev = π_θ` (same model). The synthetic responses represent the
   model's current capability ceiling.
2. Training pushes `π_θ` to prefer human responses *over its own current outputs*. This
   guarantees improvement as long as human responses are better than the model's current best.
3. At the Nash Equilibrium — when the model can no longer improve — `π_θ(y | x)` equals
   the human data distribution `p_data(y | x)`. At that point the opponent generates
   indistinguishable completions and the loss gradient vanishes.

Empirically, 3–5 iterations are sufficient for meaningful gains on 1B–7B models.

---

## How SPIN Works — Step by Step

Each SPIN **iteration** processes the full training dataset in fixed-size chunks called
**data batches** (controlled by `data_batch_size`). Within every data batch, three
atomic steps run sequentially. Each step saves its output to disk before the next step
begins, so a crash at any point resumes from the exact step boundary.

### The Inner Batch Loop

For each data batch `k` within iteration `i`:

#### Step 1 — Generate Synthetic Responses

The frozen `π_prev` model (the previous iteration's merged checkpoint, or the base
model for iteration 0) runs inference on the `data_batch_size` prompts in this chunk
to produce **synthetic responses** — these become the *rejected* side of the training
pairs.

Generation uses stochastic sampling (`do_sample=True`, temperature 0.9, top-p 0.95
by default). Prompts are sub-batched in chunks of `generation_batch_size` inside this
step to bound GPU memory (each sub-batch holds its own KV cache).

**Output:** `synthetic/iter_{i}_batch_{k:06d}_synth.jsonl`
```json
{"prompt": "...", "response": "...(human)...", "synthetic_response": "...(model)..."}
```

`π_prev` is moved to GPU for this step, then back to CPU to free GPU for training.

#### Step 2 — Score Chosen and Rejected Under `π_prev`

The same frozen `π_prev` computes **reference log-probabilities** for both sides:

```
ref_chosen_logp   = log π_prev(y_human     | prompt)
ref_rejected_logp = log π_prev(y_synthetic | prompt)
```

These scalars anchor the SPIN margin. Storing them per-batch means the trainer never
needs a second model in memory during the training loop.

**Output:** `synthetic/iter_{i}_batch_{k:06d}_logprobs.jsonl`

#### Step 3 — Train `π_θ` on This Batch

With `π_prev` back on CPU, the trainable model `π_θ` is trained on just this batch's
`data_batch_size` rows using the pre-computed logprobs:

- If `use_lora=True` (default), all base weights stay frozen and fresh LoRA adapter
  matrices are applied. Only ~0.5–2% of parameters are trainable.
- If `use_lora=False`, all parameters are updated (full fine-tuning).

The `SPINTrainer` runs `num_epochs_per_iteration` epochs on this batch. Every step
computes the SPIN margin from two forward passes (chosen and rejected), applies the
loss, and backpropagates.

After training, LoRA adapters are **merged** into the base weights and the merged
model is saved to disk. The next batch begins with fresh LoRA adapters on top of the
improved base — the base model accumulates improvements batch by batch.

**Output:** `checkpoints/iter_{i}/batch_{k:06d}/` + `.done` sentinel

A tokenized dataset cache is also written to `synthetic/iter_{i}_batch_{k:06d}_tokenized.pt`
so that the `SPINDataset` construction cost is paid only once per batch (reused on resume).

### Dataset Order

The full dataset is iterated in **fixed index order** — no shuffling. This ensures
every run (including resumes) processes identical data in an identical order.

### Iteration Completion

After all batches finish, the final batch's merged model is copied to `iter_{i}/`,
the tokenizer is saved, and an iteration-level `.done` sentinel is written.

### Crash-Safe Resumption

At startup, `_find_start_batch()` scans batch checkpoints in order and returns the
first batch where any of the three steps is incomplete. The model for that batch is
loaded from the previous batch's checkpoint directory. Within a batch's training step,
`get_last_checkpoint()` additionally detects a partially-saved HF Trainer checkpoint
to resume mid-epoch if the process was killed during backpropagation.

---

## SPIN vs Standard Fine-Tuning

| Property | Standard SFT | SPIN |
|----------|-------------|------|
| **Training signal** | Maximize log P(human response) | Maximize margin between human and model-generated responses |
| **Data requirement** | Human-labelled (prompt, response) pairs | Same dataset, no new labels needed |
| **Reward model** | Not required | Not required |
| **Human preference labels** | Not required | Not required |
| **Catastrophic forgetting** | High risk at high LR | Mitigated: gradient flows through the *difference* from `π_prev`, keeping the model close to its prior |
| **Training stability** | Generally stable | Requires careful λ tuning; `logistic` loss is most stable |
| **Convergence** | Converges to fit the data distribution | Converges to Nash equilibrium with the human data distribution |
| **Iterative improvement** | One-shot | Multi-round; each round targets a harder opponent |
| **Memory cost** | 1× model | Reference log-probs pre-computed, so still ~1× model during training |
| **Compute cost** | 1 forward + 1 backward per step | 2 forwards + 1 backward per step (chosen + rejected) |

### Key Advantage: Self-Improvement Without New Data

SPIN's fundamental advantage over SFT is that it never requires new annotations. The
dataset used for SFT can be repurposed directly. The model learns from the **gap**
between its own outputs and human quality, which is a richer signal than cross-entropy
alone. As the model improves, the gap narrows — but each new iteration's opponent is
stronger, keeping training productive.

### Key Advantage: Reference Regularization

The `π_prev` reference model acts as an implicit regulariser. The margin formulation
rewards the model for improving *relative to its previous self*, rather than driving it
to assign maximum absolute probability to human responses (which can cause reward
hacking or mode collapse in pure SFT).

### Key Limitation vs RLHF/DPO

SPIN does not use human preference rankings (A is better than B). It can only learn
from the single dimension of *human vs model* quality. For fine-grained preference
alignment (e.g. teaching specific stylistic preferences, safety behaviour, or complex
trade-offs), RLHF or DPO with preference-labelled data is still superior.

### Other LLM Training Techniques

The table below places SPIN within the broader landscape of techniques used to train
and align language models. They are grouped by the type of signal they require.

#### Overview

| Technique | Signal required | Reward model? | Iterative? | Primary goal |
|-----------|----------------|---------------|-----------|--------------|
| **Pre-training** | Unlabelled text | No | No | General language modelling |
| **Continued pre-training** | Domain text | No | No | Domain adaptation |
| **SFT / Instruction tuning** | (prompt, response) pairs | No | No | Follow instructions |
| **SPIN** *(this repo)* | Same SFT dataset | No | Yes | Close gap to human data |
| **RLHF + PPO** | Human preference rankings | Yes | Yes | Fine-grained alignment |
| **DPO** | Preference pairs (chosen > rejected) | No (implicit) | No | Preference alignment without RL |
| **IPO** | Preference pairs | No | No | Avoid DPO overfitting |
| **KTO** | Binary signals (good / bad) | No | No | Alignment from unpaired feedback |
| **ORPO** | Preference pairs | No | No | Combined SFT + preference in one loss |
| **SimPO** | Preference pairs | No | No | Reference-free preference optimisation |
| **GRPO** | Verifiable rewards (math, code) | Optional | Yes | Reasoning via group-relative rewards |
| **RLAIF** | AI-generated preference labels | No (AI critic) | Optional | Scalable alignment without human raters |
| **Constitutional AI** | AI self-critique + revision | No | Yes | Safety and helpfulness via principles |
| **Distillation** | Teacher model outputs | No | No | Compress larger model into smaller one |

---

#### Pre-training

**What it does:** Pre-training teaches the model the statistical structure of human
language by having it predict the next word in billions of sentences drawn from the
internet, books, and code. The model starts with random weights and receives no
human guidance — it simply sees a stream of text and is penalised whenever it assigns
low probability to the actual next token. Over trillions of such predictions the model
is forced to internalise grammar, facts, reasoning patterns, and writing styles in
order to keep its error low. By the end, it can continue any text plausibly, but it
has no concept of following instructions or behaving safely — it will equally
fluently complete a recipe, a phishing email, or a calculus proof.

The training signal is **next-token prediction** (also called causal language
modelling). Given all previous tokens in a sequence, the model outputs a probability
distribution over its vocabulary and is trained to assign maximum probability to the
token that actually came next:

```
L_PT = − Σ_t log P(token_t | token_0…token_{t-1})
```

Pre-training is compute-intensive (thousands of GPU-days for 7B+ models) and requires
no annotation. The result is a model that can continue text but has no instruction-
following or safety behaviour. All subsequent techniques fine-tune a pre-trained base.

---

#### Continued Pre-training (CPT)

**What it does:** A general-purpose base model has seen a little of everything but is
not particularly expert at anything. CPT re-runs the same next-token-prediction
training loop — without any new type of supervision — but feeds the model a curated
corpus of domain-specific text (e.g. medical literature, legal contracts, Python
repositories). Because the model already understands language, it converges quickly to
the domain's vocabulary, conventions, and reasoning patterns. The result is a model
that retains its general capabilities but produces far more accurate and idiomatic
outputs within the target domain. CPT is almost always followed by SFT to add
instruction-following behaviour.

The training objective is identical to pre-training (next-token prediction); only the
data distribution changes. A general base model is used as the starting point rather
than random weights, so far fewer tokens are needed to achieve domain adaptation.

---

#### Supervised Fine-Tuning (SFT) and Instruction Tuning

**What it does:** A pre-trained model can predict text but does not know how to answer
questions, follow instructions, or behave helpfully. SFT bridges this gap by showing
the model thousands of worked examples: a human writes a prompt and a high-quality
response, and the model is trained to reproduce that response given the prompt. The
loss function is still next-token prediction, but now it is computed only over the
response tokens — the prompt tokens are masked out so the model learns to generate
answers, not re-generate the question. After SFT the model has shifted from
"continues any text" to "answers questions and follows instructions in the style of
the training demonstrations".

**Instruction tuning** is SFT applied specifically to instruction-following datasets
(e.g. FLAN, Alpaca, UltraChat) — the mechanics are identical but the data is
structured as `(instruction, response)` pairs covering diverse task types. The two
terms are used interchangeably.

The loss penalises the model for every token in the response where its predicted
probability was not highest on the ground-truth token:

```
L_SFT = − Σ_t log P(token_t | prompt, response_tokens_0…t-1)
```

Only response positions contribute to the gradient (prompt positions are masked with
`-100` in the labels tensor). This is the training regime implemented in
[spin_dataset.py](spin_dataset.py) for the human-response side of each SPIN example.

---

#### RLHF + PPO

**What it does:** SFT teaches the model to imitate demonstrations, but imitation has
a ceiling — the model cannot learn to be *better than* the training examples, only to
reproduce their style. RLHF breaks through that ceiling by letting the model explore
freely and then rewarding it for outputs that humans actually prefer. Human raters
compare pairs of responses (`A vs B`) and record which one is better. A separate
**reward model** is trained on these rankings so it can score any new response with a
single number. The **policy model** (the LLM being aligned) is then trained with
Proximal Policy Optimization (PPO) — a reinforcement learning algorithm — to generate
responses that maximise the reward model's score. A KL-divergence penalty keeps the
policy from drifting so far from the SFT checkpoint that it produces incoherent text
purely to fool the reward model (reward hacking).

The result is a model that can optimise for subtle human preferences — tone, safety,
helpfulness, refusal of harmful requests — that no fixed demonstration dataset could
fully capture. GPT-4, Claude, and Gemini all use variants of this pipeline.

Reinforcement Learning from Human Feedback, introduced by OpenAI (InstructGPT, 2022).
The pipeline has three stages:

1. **SFT** — fine-tune the base model on demonstration data to obtain a well-behaved
   starting policy.
2. **Reward model training** — collect human preference rankings (`A > B` for the same
   prompt) and train a separate reward model `r(prompt, response) → scalar` that
   learns to assign higher scores to responses humans prefer.
3. **PPO** — use Proximal Policy Optimisation (an RL algorithm) to update the policy
   model to maximise the reward model's score, subject to a KL-divergence penalty
   against the SFT model to prevent reward hacking.

```
L_PPO = E[r(x, y)] − β · KL(π_θ || π_SFT)
```

The PPO clip ratio bounds how large a single policy update can be, trading off
sample efficiency for stability. In practice RLHF training requires careful balancing
of reward scale, KL weight β, and clip ratio — small misconfigurations produce
incoherent or sycophantic outputs.

**Advantages:** Can optimise for arbitrary reward signals; has produced the most
aligned models to date (GPT-4, Claude, Gemini all use variants).

**Disadvantages:** Requires a reward model (expensive to train), two models in memory
during RL, is notoriously unstable (sensitive to PPO clip ratio, KL penalty β, and
reward model quality), and prone to reward hacking if the reward model is imperfect.

---

#### DPO — Direct Preference Optimization

**What it does:** DPO solves the same problem as RLHF — teaching the model to prefer
better responses — but sidesteps the reward model and RL loop entirely. The key
insight is that the optimal RLHF policy has a closed-form relationship to any reward
function: if you know what policy RLHF would converge to, you can write the reward
implicitly in terms of the policy's own log-probabilities. DPO substitutes this
implicit reward into the preference-ranking loss (the Bradley-Terry model), simplifying
everything into a single regression objective that trains the policy directly.

In practice, you give the model a dataset of preference pairs — for the same prompt,
a preferred response `y_w` and a dispreferred response `y_l`. DPO's loss increases the
probability the model assigns to `y_w` relative to a frozen reference model, while
simultaneously decreasing the probability it assigns to `y_l`. The frozen reference
model (the SFT checkpoint) acts as an implicit KL penalty, preventing the model from
drifting too far. No reward model is trained, no RL updates are performed, and only
one model is needed in memory at training time (the reference model's log-probs can be
pre-computed and stored, mirroring the approach used by SPIN).

DPO (Rafailov et al., 2023) re-derives the RLHF objective in closed form, eliminating
the need for a separate reward model or RL training loop. The optimal RLHF policy
implies an implicit reward:

```
r*(x, y) = β · log [π*(y|x) / π_ref(y|x)] + β · log Z(x)
```

Substituting this into the Bradley-Terry preference model and simplifying yields the
DPO loss directly in terms of the policy:

```
L_DPO = −log σ( β · [log π_θ(y_w|x) − log π_ref(y_w|x)]
                   − β · [log π_θ(y_l|x) − log π_ref(y_l|x)] )
```

where `y_w` is the preferred (winning) response and `y_l` is the dispreferred
(losing) response. A frozen reference model `π_ref` (the SFT checkpoint) anchors
the KL penalty implicitly.

**Relationship to SPIN:** The DPO loss and the SPIN logistic loss are structurally
similar. The key differences are:

| | DPO | SPIN |
|---|---|---|
| `y_l` source | Human-labelled dispreferred response | Model-generated synthetic response |
| `y_w` source | Human-labelled preferred response | Ground-truth human response |
| `π_ref` | Fixed SFT checkpoint (never updated) | Updated each iteration (`π_prev`) |
| Iterative? | No — single training pass | Yes — multi-round |
| Annotation | Requires preference pairs | Only requires SFT-style data |

**Advantages over RLHF:** No reward model, no RL instability, single training stage.

**Disadvantages:** Requires preference-labelled data (`A > B` pairs); performance is
sensitive to the distribution of the reference model; can overfit to the preference
format without generalising.

---

#### IPO — Identity Preference Optimization

**What it does:** IPO is a direct fix for a specific failure mode in DPO. Once DPO
has learned to separate `y_w` from `y_l` by a large margin, the sigmoid loss saturates
— its gradient approaches zero — so the model stops receiving a meaningful training
signal. Without gradient pressure, the model's weight updates become noisy and it may
drift toward deterministic outputs (always outputting the single highest-probability
token), which reduces diversity and causes it to fail on prompts that require nuanced
or varied responses.

IPO replaces DPO's saturating sigmoid loss with a squared-error loss that has a fixed
target margin of `1/2β`. Regardless of how large the current margin already is, the
squared-error term produces a non-zero gradient pushing the margin toward exactly the
target — not toward infinity. This keeps training well-behaved throughout and prevents
the model from collapsing to a near-deterministic policy.

IPO (Azar et al., 2023) addresses an overfitting failure mode in DPO: because DPO
uses a logistic (sigmoid) loss, the gradient approaches zero as the margin between
`y_w` and `y_l` grows large. The model can assign near-infinite log-probability ratio
to the preferred response while still minimising the loss — equivalent to deterministic
mode collapse.

IPO replaces the logistic loss with a squared-error loss that penalises any margin
beyond 1/2:

```
L_IPO = ( [log π_θ(y_w|x)/π_ref(y_w|x)] − [log π_θ(y_l|x)/π_ref(y_l|x)] − 1/2β )²
```

This gives a non-vanishing gradient even when the model already separates the two
responses well, preventing deterministic collapse.

---

#### KTO — Kahneman-Tversky Optimization

**What it does:** DPO and its variants all require preference *pairs* — for every
training example you need two responses to the same prompt and a label saying which is
better. This means you cannot use datasets that contain only good examples (e.g. a
curated answer corpus) or only bad examples (e.g. a toxicity dataset) independently.
KTO removes this constraint by treating each response in isolation: every example is
simply stamped `desirable` (the model should be more likely to produce this) or
`undesirable` (the model should be less likely to produce this), with no pairing
between them.

The loss is shaped by Kahneman and Tversky's prospect theory from behavioural
economics, which observes that humans feel losses more acutely than equivalent gains.
KTO mirrors this asymmetry: it penalises the model more for moving toward undesirable
outputs than it rewards it for moving toward desirable ones. A per-prompt KL term
(computed against a reference batch from the training data) serves as a baseline,
analogous to a value function in RL, so the model is rewarded or penalised relative
to what it would normally produce for that prompt rather than in absolute terms.

KTO (Ethayarajh et al., 2024) uses **unpaired binary signals**: each example is simply
labelled `desirable` or `undesirable` independently, with no pairing required.

```
L_KTO = 1 − σ( z_ref(x) − β · KL(π_θ(y|x) || π_ref(y|x)) )   if y is desirable
       1 − σ( β · KL(π_θ(y|x) || π_ref(y|x)) − z_ref(x) )     if y is undesirable
```

where `z_ref(x)` is a per-prompt baseline estimated from a reference batch.

**Key advantage:** Works with unpaired positive-only or negative-only examples — any
dataset where responses are tagged as good or bad, without needing a ranked pair.

---

#### ORPO — Odds Ratio Preference Optimization

**What it does:** DPO requires a frozen reference model to anchor the KL penalty,
which means you need either a second model in memory or pre-computed reference
log-probabilities stored on disk. ORPO eliminates the reference model entirely by
building the preference signal into the SFT loss itself.

For each training example, ORPO runs a single forward pass that sees both the
preferred response `y_w` and the dispreferred response `y_l`. Two terms are combined:
a standard SFT cross-entropy loss that rewards the model for generating `y_w`, and an
odds-ratio term that compares how many times more likely the model is to produce `y_w`
versus `y_l`. Because the odds ratio is computed purely from the current model's
outputs — with no reference to a frozen checkpoint — the model acts as its own
implicit baseline. The higher the odds ratio, the smaller the loss; the loss is large
when the model is nearly equally likely to produce either response.

This design lets you train a base model directly to an instruction-following,
preference-aligned state in a single stage, without first running a separate SFT
phase.

ORPO (Hong et al., 2024) formalises this as:

```
L_ORPO = L_SFT + λ · L_OR
L_OR   = −log σ( log [odds_θ(y_w|x) / odds_θ(y_l|x)] )
odds_θ(y|x) = P_θ(y|x) / (1 − P_θ(y|x))
```

The SFT term maximises log-probability on the preferred response; the odds-ratio term
simultaneously penalises the dispreferred response — both computed from the same
forward pass without a reference model.

**Key advantage:** ~33% less memory than DPO (no reference model forward pass); trains
in a single stage from the base model directly to an aligned model.

---

#### SimPO — Simple Preference Optimization

**What it does:** SimPO starts from the same goal as DPO — push the model toward
preferred responses and away from dispreferred ones — but removes two components that
add complexity without consistently improving results: the frozen reference model and
the KL penalty term.

Without a reference model, SimPO scores each response by its average per-token
log-probability under the current model: a response's total log-probability divided by
its length. This length normalisation is crucial. DPO's raw log-probability sum is
biased toward shorter responses, because every additional token in a long response
adds another (negative) log-probability term to the sum — a shorter response
trivially wins even if it is worse quality. By dividing by response length, SimPO
makes the score comparable across responses of different lengths.

A configurable target margin `γ` sets the minimum gap the model must maintain between
its scores for `y_w` and `y_l` before the loss reaches zero. This margin acts like a
confidence threshold: the model is not just penalised for getting the preference wrong,
it is penalised until it has a clear, definitive preference — preventing shallow
alignment where the preferred response barely edges out the dispreferred one.

SimPO (Meng et al., 2024) formalises this as:

```
L_SimPO = −log σ( β/|y_w| · log π_θ(y_w|x) − β/|y_l| · log π_θ(y_l|x) − γ )
```

where `|y|` is the response length and `γ > 0` is a target margin.

**Key advantages:** No reference model; no KL penalty divergence; competitive with
DPO-family methods on instruction-following benchmarks at lower computational cost.

---

#### GRPO — Group Relative Policy Optimization

**What it does:** For tasks like mathematics and programming, you do not need a human
to say which of two answers is better — you can check automatically: does the code
run? does the final answer equal the ground truth? GRPO exploits this by letting the
model generate a *group* of `G` independent responses to the same prompt (typically
8–16 samples), then scoring each one with a verifiable reward function. It uses the
group's own statistics as the baseline: a response that scored above the group average
is treated as a "win" and is reinforced; one that scored below the average is
penalised. No human labels, no reward model, and no comparison between separate
prompts are required.

The training signal is computed entirely within each group. If six out of eight
sampled responses solved the problem correctly and two did not, the two failures
receive a negative advantage (pushed away from) and the six successes receive a
positive advantage (pushed toward). A KL penalty against a reference model prevents
the policy from collapsing to deterministic output by always sampling the same
response — diversity within the group is essential for the training signal to exist.

GRPO also naturally trains the model to produce chain-of-thought reasoning: by
rewarding responses that arrive at correct answers and penalising those that do not,
the model discovers that showing working (reasoning steps) reliably leads to better
final answers. This is the mechanism behind DeepSeek-R1's reasoning capability.

GRPO (DeepSeek-R1, 2024) formalises this as:

```
A_i = (r_i − mean(r)) / std(r)
L_GRPO = − Σ_i A_i · log π_θ(y_i|x) + β · KL(π_θ || π_ref)
```

Training pushes the model toward responses that scored above the group mean and away
from those that scored below — without ever constructing explicit `(y_w, y_l)` pairs.

**Key advantage:** No human labelling required for domains with verifiable rewards.
GRPO is the core training objective behind DeepSeek-R1's chain-of-thought reasoning.

**Relationship to SPIN:** Both are iterative and self-improving. SPIN uses the
previous iteration's model to generate the rejected side; GRPO uses within-batch
group statistics as the baseline. GRPO requires a verifiable reward signal; SPIN only
requires ground-truth human responses.

---

#### RLAIF — Reinforcement Learning from AI Feedback

**What it does:** RLHF's biggest bottleneck is human annotation: trained human raters
are slow, expensive, and cannot practically label millions of response pairs. RLAIF
replaces human raters with a large, capable AI model (the "critic" or "judge") that
can read two responses to the same prompt and output a preference label — at machine
speed and arbitrary scale.

The pipeline is structurally identical to RLHF. For each training prompt, the policy
model generates two candidate responses. The AI critic reads both and decides which is
better, often outputting a short reasoning chain before its verdict. These
AI-generated preferences are accumulated into a preference dataset, which is used
either to train a reward model (then PPO) or fed directly into DPO. Because the critic
runs autonomously, millions of preference labels can be generated overnight.

The central tradeoff is that the model being trained learns to satisfy the critic, not
actual humans. If the critic has systematic biases — preferring longer responses,
certain political framings, or confident-sounding language regardless of accuracy —
the trained model will inherit those biases. RLAIF is therefore most reliable when the
critic model is substantially more capable than the model being trained, so the critic
can make genuine quality judgements rather than superficial pattern matches.

RLAIF (Lee et al., 2023; Bai et al., 2022 — Constitutional AI) mirrors RLHF:

1. For each prompt, sample two responses from the policy model.
2. Ask a large critic model (e.g. Claude, GPT-4) to choose the better response and
   explain its reasoning.
3. Use the AI-generated preferences to train a reward model (or feed directly into DPO).
4. Fine-tune the policy model with PPO or DPO using this reward signal.

**Key advantage:** Scales to millions of examples without human rater bottlenecks.

**Key disadvantage:** The policy model can overfit to the biases of the AI critic,
inheriting its blind spots (style preferences, sycophancy, political biases) rather
than learning human values directly.

---

#### Constitutional AI (CAI)

**What it does:** Standard RLAIF lets the AI critic judge responses freely, which
means the alignment criteria live implicitly inside the critic's weights — they are
opaque, hard to audit, and impossible to change without retraining the critic. CAI
makes the alignment criteria *explicit* by writing them down as a human-readable list
of principles (the "constitution"): for example, "prefer responses that are helpful",
"prefer responses that avoid discriminatory content", "prefer responses that are honest
even when the answer is uncertain".

The training pipeline runs in two stages. In the **supervised stage**, the model is
shown its own initial response to a potentially harmful or unhelpful prompt and is
asked to critique that response against each constitutional principle in turn, then
revise it. The chain of (critique, revision) pairs is iterated until the response
satisfies all principles. The final revised response becomes supervised fine-tuning
data, teaching the model to self-correct toward the constitutional ideals.

In the **RL stage**, a separate AI feedback model — guided by the same constitution —
compares response pairs and generates preference labels. These labels train a reward
model, which is then used in PPO to further align the policy. Because both stages
reference the same written constitution, the alignment criteria are auditable: if the
model behaves incorrectly in production, practitioners can read the constitution to
understand why, and edit a principle to change future behaviour without rebuilding
the entire pipeline.

Constitutional AI (Anthropic, 2022) is a specific RLAIF variant. The two stages are:

1. **Supervised stage** — the model generates a response, critiques it against each
   constitutional principle in sequence, and revises it. The final revised response
   becomes SFT training data.
2. **RL stage (RLAIF)** — a separate AI feedback model scores response pairs according
   to the same constitution; these scores train a preference model used in PPO.

**Key advantage:** Makes the alignment criteria legible and modifiable — changing the
constitution changes the model's behaviour without retraining from scratch.

---

#### Knowledge Distillation

**What it does:** A large, expensive model (the "teacher") already contains vast
knowledge, but it is too slow or too memory-hungry to deploy. Distillation trains a
smaller, cheaper model (the "student") to reproduce the teacher's outputs — not just
its final answers, but the full probability distribution it assigns to every possible
next token. This richer signal teaches the student about the teacher's uncertainty
and near-misses, not just its top choice.

Consider a question with the correct answer "Paris". A one-hot training label says
only "Paris is right, everything else is equally wrong". The teacher's output
distribution might say "Paris: 82%, Lyon: 4%, Marseille: 2%, …", which tells the
student that French cities are plausible — a signal invisible in the hard label. By
minimising the divergence between its distribution and the teacher's, the student
learns a compressed but faithful approximation of the teacher's knowledge.

There are two main variants depending on whether the teacher's internal probability
distribution is accessible:

**Logit distillation** — the student minimises KL divergence between its output
distribution and the teacher's full probability distribution over the vocabulary:

```
L_KD = KL( p_teacher(·|x) || p_student(·|x) )
```

This requires access to the teacher's raw logits (available when both models run
locally). Soft targets carry richer information than one-hot labels because the
teacher assigns small but nonzero probability to semantically related tokens — a
signal that is completely invisible in hard labels.

**Sequence-level distillation** — the teacher generates complete response strings and
the student is trained on these synthetic responses via SFT (next-token prediction).
This is simpler and scales to any teacher, including API-only models where logits are
not exposed. A large fraction of open-source instruction models (Alpaca, Vicuna, Orca)
are distilled this way from GPT-3.5/GPT-4 — their training data is the teacher's
generated text, not human demonstrations.

**Relationship to SPIN:** SPIN can be viewed as a form of self-distillation where the
teacher is the model from the previous iteration rather than a separate larger model.
The synthetic responses generated by `π_prev` play the role of teacher outputs.
Unlike cross-model distillation, no external model is required — the student improves
by repeatedly competing against its own prior self.

---

#### Technique Selection Guide

```
Do you have human-annotated preference pairs (A > B)?
  ├── Yes → DPO / IPO / SimPO  (no reward model; stable; recommended default)
  │         RLHF + PPO          (if budget allows; strongest alignment signal)
  └── No
       ├── Do you have binary good/bad signals (unpaired)?
       │     └── Yes → KTO
       ├── Do you want SFT + preference in one pass?
       │     └── Yes → ORPO
       ├── Can correctness be verified automatically (math, code)?
       │     └── Yes → GRPO
       ├── Do you only have SFT-style (prompt, response) data?
       │     └── Yes → SPIN  (iterative self-improvement; no new data needed)
       └── Do you want to compress a larger model?
             └── Yes → Distillation (logit or sequence-level)
```

