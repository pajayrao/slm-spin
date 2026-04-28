# slm-spin

Implementation of the SPIN (Self-Play Fine-Tuning) algorithm for Small Language Models, optimised for consumer and research GPUs.

## Overview

SPIN iteratively improves a language model by training it to prefer human responses over responses it generates itself. Each iteration:

1. **Generate** — the current model produces synthetic responses for each training prompt (the "opponent").
2. **Compute ref log-probs** — the frozen previous-iteration model scores both the human and synthetic responses.
3. **Train** — the model is fine-tuned to increase the probability of human responses relative to its own synthetic ones, using a contrastive SPIN loss.

After _N_ iterations the final checkpoint is written to `checkpoints_dir/iter_{N-1}/`.

## Features

- **LoRA / full fine-tuning** — LoRA is enabled by default, reducing trainable parameters by 100×.
- **Memory-split backward pass** — chosen and rejected sequences are backpropagated separately so peak activation memory equals `max(chosen, rejected)` rather than `chosen + rejected`.
- **Growing curriculum** — synthetic data from all past iterations can be accumulated for richer training signal.
- **Crash-safe resumption** — `.done` sentinels and JSONL caches let you restart after a kill without regenerating or re-training anything.
- **TensorBoard integration** — per-iteration and global cross-iteration logs, PR curves, embedding projector, parameter histograms, and optional profiler traces.
- **Multiple loss functions** — logistic (default), hinge, correlation, exponential.
- **RMSProp and AdamW** — RMSProp uses ~2 GB less GPU memory per ~1 B-parameter model.
- **Benchmark evaluation** — six standard LLM benchmarks evaluated directly from checkpoints, with iteration-over-iteration comparison and TensorBoard logging.

## Installation

```bash
pip install -r requirements.txt
```

Key runtime dependencies: `transformers`, `peft`, `torch`, `datasets`, `accelerate`, `tensorboard`.

A CUDA-capable GPU is required for practical training. The code targets CUDA 13.0 / PyTorch 2.11.

## Quick Start

```bash
python main.py \
  --model_name_or_path microsoft/harrier-oss-v1-0.6b \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --train_split train_sft \
  --num_iterations 3 \
  --synthetic_examples_per_iteration 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --output_dir ./runs/my_run
```

All `SPINConfig` fields are exposed as CLI flags — pass any field as `--field_name value`.

### Local dataset

```bash
python main.py \
  --data_path ./my_data.jsonl \
  --model_name_or_path /path/to/local/model \
  --num_iterations 3
```

JSONL files must contain records with `prompt` and `response` keys, **or** a `messages` list of `{"role": ..., "content": ...}` dicts.

## Configuration Reference

All options live in [spin_config.py](spin_config.py). The most important ones:

| Flag | Default | Description |
|------|---------|-------------|
| `model_name_or_path` | `microsoft/harrier-oss-v1-0.6b` | HuggingFace Hub ID or local checkpoint path |
| `dataset_name` | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset (overridden by `data_path`) |
| `num_iterations` | `5` | Number of SPIN outer loops |
| `synthetic_examples_per_iteration` | `128` | Prompts to generate per iteration; `0` = full dataset |
| `accumulate_previous_synthetic` | `True` | Include synthetic data from prior iterations |
| `lambda_initial` | `0.1` | SPIN loss scale λ for all but the last iteration |
| `lambda_final_iteration` | `5.0` | λ for the final iteration (stronger alignment push) |
| `loss_type` | `logistic` | `logistic` \| `hinge` \| `correlation` \| `exponential` |
| `use_lora` | `True` | Enable LoRA (strongly recommended on small GPUs) |
| `lora_r` | `16` | LoRA rank |
| `optimizer` | `rmsprop` | `rmsprop` (less memory) \| `adamw` |
| `per_device_train_batch_size` | `6` | Reduce to `1` on an 8 GB GPU |
| `gradient_accumulation_steps` | `32` | Compensate for small batch size |
| `learning_rate` | `5e-7` | Peak LR for early iterations |
| `learning_rate_late` | `1e-7` | LR from `late_lr_start_iteration` onward |
| `max_length` | `1024` | Max tokens (prompt + response) during training |
| `bf16` | `True` | bfloat16 mixed precision (Ampere+ GPU required) |
| `gradient_checkpointing` | `False` | Recompute activations to save ~10× memory at ~33% compute cost |
| `output_dir` | `./spin_outputs` | Root directory for all outputs |

### SPIN loss

The loss operates on a margin per training example:

```
margin = λ × [(log π_θ(chosen) − log π_ref(chosen)) − (log π_θ(rejected) − log π_ref(rejected))]
```

A positive margin means the model has improved more on the human response than on the synthetic one. The `loss_type` maps this margin to a scalar:

| `loss_type` | Formula | Notes |
|-------------|---------|-------|
| `logistic` | `softplus(−margin)` | Smooth, never saturates — recommended |
| `hinge` | `relu(1 − margin)` | Zero loss once margin > 1 |
| `correlation` | `1 − margin` | Constant gradient, easiest to tune |
| `exponential` | `exp(−margin)` | Aggressive on negative margins; can be unstable |

## Evaluation

[evaluate.py](evaluate.py) evaluates every trained checkpoint against six standard LLM benchmarks and logs iteration-over-iteration comparisons to TensorBoard. It requires no `lm_eval` dependency — scoring is implemented directly with HuggingFace `transformers`.

### Benchmarks

| Benchmark | Metric | Shots | Scoring method |
|-----------|--------|------:|----------------|
| ARC-Challenge | acc_norm | 25 | Length-normalised log-likelihood over 4 choices |
| TruthfulQA MC2 | mc2 | 0 | Softmax probability mass on all correct choices |
| Winogrande | acc | 5 | Log-likelihood of each fill option in context |
| GSM8k | acc | 5 | Greedy generation + exact numeric string match |
| HellaSwag | acc_norm | 10 | Length-normalised log-likelihood over 4 endings |
| MMLU | acc | 5 | Log-likelihood of single letter (A/B/C/D) continuation |

Shot counts match the Open LLM Leaderboard v1 defaults so results are directly comparable to published numbers.

**acc_norm** divides each choice's log-likelihood by its character length before picking the winner — this prevents the model from trivially preferring shorter answers.

**mc2** (TruthfulQA) applies softmax across all choices and sums the probability mass landing on the subset of correct answers. A score of 1.0 means the model assigned all probability to true statements.

### Running evaluation

```bash
# Evaluate all iter_* checkpoints found in the default directory
python evaluate.py

# Evaluate a specific subset of iterations
python evaluate.py --iters iter_0 iter_2 iter_4

# Smoke test with 50 examples per task (do not use for real benchmarks)
python evaluate.py --limit 50

# Override shot counts for specific tasks
python evaluate.py --n-shots arc_challenge=10 gsm8k=3

# Custom checkpoint and output directories
python evaluate.py \
  --checkpoints-dir ./runs/my_run/checkpoints \
  --output-dir      ./runs/my_run/eval_results \
  --tensorboard-dir ./runs/my_run/tensorboard/eval
```

### Evaluation output

For each iteration the evaluator prints a live progress line per task (n_shot, example count, elapsed seconds), followed by an iteration summary with ▲/▼ direction indicators:

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

After all iterations a formatted comparison table is printed:

```
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
| Iteration | Arc  | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU  | Avg%  | ΔAvg  | Status |
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
| iter_0   | 40.80 | 50.10      | 62.30     | 18.50 | 71.20     | 46.00 | 48.15 |   NA  |        |
| iter_1   | 42.15 | 51.30      | 61.90     | 19.20 | 72.40     | 47.30 | 49.04 | ▲+0.89| ★ BEST |
+----------+-------+------------+-----------+-------+-----------+-------+-------+-------+--------+
```

Written files:

```
eval_results/
├── iter_0.parsed.json          # Raw scores for iteration 0
├── iter_1.parsed.json
├── comparative_summary.txt     # TSV table + per-iteration narrative (paste into spreadsheet)
└── comparative_summary.json    # Machine-readable version of the full results list
```

### Evaluation TensorBoard panels

```bash
tensorboard --logdir ./spin_outputs/tensorboard
```

The evaluation run writes to `tensorboard/eval_compare/` and adds the following panels:

| Panel | Tags | Description |
|-------|------|-------------|
| **Custom Scalars → Evaluation** | `eval/average`, `eval/best_so_far_average` | Average score and running best across iterations |
| **Custom Scalars → Evaluation** | `eval/tasks/<name>` | Per-task score for each iteration |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/<name>` | Per-task score change from preceding iteration |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/improvement_rate` | Fraction of tasks that improved (0–1) |
| **Custom Scalars → Delta vs Previous** | `compare_vs_prev/improved_task_count`, `…/declined_task_count` | Count of tasks that improved / declined |
| **Text → eval/scorecard** | — | Markdown table of scores + deltas, one card per iteration |
| **Text → eval/best_iteration** | — | Note written each time a new best average is reached |
| **Text → eval/run_config** | — | Shot counts, device, directories — written once at step 0 |

## Output Structure

```
spin_outputs/
├── config.json                      # Snapshot of SPINConfig used for this run
├── tb_global/                       # Cross-iteration TensorBoard run (training metrics)
├── tb_profile/                      # PyTorch profiler traces (if enabled)
├── synthetic/
│   ├── iter_0.jsonl                 # Synthetic prompt/response pairs from iteration 0
│   └── iter_1.jsonl
├── checkpoints/
│   ├── iter_0/
│   │   ├── config.json              # Model config
│   │   ├── model.safetensors        # Merged weights (LoRA adapters merged in)
│   │   ├── tokenizer_config.json
│   │   ├── .done                    # Sentinel written after full save
│   │   └── tb_logs/                 # Per-iteration TensorBoard run (training)
│   └── iter_1/
│       └── ...
├── eval_results/
│   ├── iter_0.parsed.json           # Per-task benchmark scores for this checkpoint
│   ├── iter_1.parsed.json
│   ├── comparative_summary.txt      # Human-readable TSV + narrative comparison
│   └── comparative_summary.json     # Full results list (all iterations)
└── tensorboard/
    └── eval_compare/                # Evaluation TensorBoard run
```

The final model is at `checkpoints/iter_{num_iterations-1}/`.

## Resuming After a Crash

No flags are needed. On restart, the script:

1. Scans `checkpoints_dir` for the last iteration that wrote a `.done` sentinel.
2. Reloads cached synthetic JSONL files for completed iterations.
3. Calls `get_last_checkpoint()` inside the current iteration directory to resume mid-training if a HuggingFace checkpoint exists.

## TensorBoard (training)

```bash
tensorboard --logdir ./spin_outputs
```

The `tb_global/` run plots metrics across all iterations on a single x-axis. Individual `iter_N/tb_logs/` runs show per-step detail for each iteration.

Available panels (depending on config flags):

- **Scalars** — loss, margin mean/std, win rate, log-probs, KL from ref, learning rate
- **PR Curves** — alignment accuracy per epoch
- **Projector** — token embedding shift across iterations (PCA / UMAP / t-SNE)
- **Histograms** — per-layer weight and gradient distributions
- **Trace** and **Memory** — PyTorch profiler output

## Memory Tips for Small GPUs (≤ 16 GB)

- Set `--use_lora True` (default) — cuts gradient/optimizer memory by ~10–100×.
- Lower `--per_device_train_batch_size 1` and raise `--gradient_accumulation_steps 64`.
- Enable `--gradient_checkpointing True` — trades ~33% compute for ~10× less activation memory.
- Use `--optimizer rmsprop` — saves ~2 GB vs AdamW on a 1 B-parameter model.
- Reduce `--max_length 512` and `--max_prompt_length 256`.
- Set `--generation_batch_size 4` — each beam holds its own KV cache during generation.
- Set `--bf16 True` (default on Ampere+) or `--fp16 True` on older GPUs.

## Project Structure

| File | Description |
|------|-------------|
| [main.py](main.py) | Entry point — outer SPIN loop, resume logic, orchestration |
| [evaluate.py](evaluate.py) | Benchmark evaluation — six tasks, iteration comparison, TensorBoard logging |
| [spin_config.py](spin_config.py) | `SPINConfig` dataclass — all hyperparameters with inline docs |
| [spin_trainer.py](spin_trainer.py) | `SPINTrainer` and `RMSPropSPINTrainer` — loss and memory-split backward |
| [spin_dataset.py](spin_dataset.py) | `SPINDataset` — pre-tokenises chosen/rejected pairs |
| [spin_data_collator.py](spin_data_collator.py) | `SPINDataCollator` — pads and batches chosen/rejected tensors |
| [utils.py](utils.py) | Model loading, tokenisation, generation, LoRA merge, arg parsing |
| [trainer_callback/](trainer_callback/) | TensorBoard, profiler, memory probe, and iteration summary callbacks |
