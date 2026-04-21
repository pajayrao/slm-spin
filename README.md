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

## Output Structure

```
spin_outputs/
├── config.json                    # Snapshot of SPINConfig used for this run
├── tb_global/                     # Cross-iteration TensorBoard run
├── tb_profile/                    # PyTorch profiler traces (if enabled)
├── synthetic/
│   ├── iter_0.jsonl               # Synthetic prompt/response pairs from iteration 0
│   └── iter_1.jsonl
└── checkpoints/
    ├── iter_0/
    │   ├── config.json            # Model config
    │   ├── model.safetensors      # Merged weights (LoRA adapters merged in)
    │   ├── tokenizer_config.json
    │   ├── .done                  # Sentinel written after full save
    │   └── tb_logs/               # Per-iteration TensorBoard run
    └── iter_1/
        └── ...
```

The final model is at `checkpoints/iter_{num_iterations-1}/`.

## Resuming After a Crash

No flags are needed. On restart, the script:

1. Scans `checkpoints_dir` for the last iteration that wrote a `.done` sentinel.
2. Reloads cached synthetic JSONL files for completed iterations.
3. Calls `get_last_checkpoint()` inside the current iteration directory to resume mid-training if a HuggingFace checkpoint exists.

## TensorBoard

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
| [spin_config.py](spin_config.py) | `SPINConfig` dataclass — all hyperparameters with inline docs |
| [spin_trainer.py](spin_trainer.py) | `SPINTrainer` and `RMSPropSPINTrainer` — loss and memory-split backward |
| [spin_dataset.py](spin_dataset.py) | `SPINDataset` — pre-tokenises chosen/rejected pairs |
| [spin_data_collator.py](spin_data_collator.py) | `SPINDataCollator` — pads and batches chosen/rejected tensors |
| [utils.py](utils.py) | Model loading, tokenisation, generation, LoRA merge, arg parsing |
| [trainer_callback/](trainer_callback/) | TensorBoard, profiler, memory probe, and iteration summary callbacks |
