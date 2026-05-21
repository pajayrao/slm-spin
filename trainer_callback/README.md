# Trainer Callbacks

Five HuggingFace `TrainerCallback` subclasses that instrument SPIN training with TensorBoard metrics, memory monitoring, parameter statistics, and PyTorch profiling.

---

## Table of Contents

1. [TensorBoardCallbackExtended](#1-tensorboardcallbackextended)
2. [SPINIterationSummaryCallback](#2-spiniterationsummarycallback)
3. [TensorBoardParameterStatsCallback](#3-tensorboardparameterstatscallback)
4. [MemoryProbeCallback](#4-memoryprobecallback)
5. [TorchProfilerCallback](#5-torchprofilercallback)

---

## 1. TensorBoardCallbackExtended

**File:** [tensorboard_callback_extended.py](tensorboard_callback_extended.py)

Per-iteration TensorBoard logger. A new instance is created for each SPIN iteration and writes to its own subdirectory (`iter_dir/tb_logs`), enabling side-by-side comparison of iterations in TensorBoard.

### Constructor

```python
TensorBoardCallbackExtended(log_dir, cfg, log_histograms=False, tokenizer=None, iteration=0)
```

| Parameter | Description |
|---|---|
| `log_dir` | TensorBoard log directory for this iteration |
| `cfg` | `SPINConfig` controlling which optional metrics are enabled |
| `log_histograms` | Write weight/gradient histogram events (produces large files) |
| `tokenizer` | Used to decode token IDs as human-readable labels in the PROJECTOR tab |
| `iteration` | SPIN iteration index — recorded in hparam entries for cross-run comparison |

---

### Metrics Written

#### `train/*` — logged on every `on_log` call (each training step)

| Metric | Description | Calculation |
|---|---|---|
| `train/loss` | SPIN margin loss | Passed directly from `SPINTrainer.training_step` logs |
| `train/margin_mean` | Mean SPIN margin across the batch | `mean(log π_θ(y_w) − log π_ref(y_w) − log π_θ(y_l) + log π_ref(y_l))` per batch |
| `train/margin_std` | Std-dev of SPIN margin | Standard deviation of per-example margins; high std = uneven alignment signal |
| `train/win_rate` | Fraction of batch examples where margin > 0 | `count(margin > 0) / batch_size` |
| `train/pi_chosen_logp` | Mean log-prob of human response under π_θ | `mean(log π_θ(y_w))` — should increase as the model learns |
| `train/pi_rejected_logp` | Mean log-prob of synthetic response under π_θ | `mean(log π_θ(y_l))` — should decrease relative to chosen |
| `train/ref_chosen_logp` | Mean log-prob of human response under π_ref (frozen) | `mean(log π_ref(y_w))` — constant; useful as baseline |
| `train/ref_rejected_logp` | Mean log-prob of synthetic response under π_ref (frozen) | `mean(log π_ref(y_l))` — constant; useful as baseline |
| `train/logp_gap` | Log-prob gap between human and synthetic responses under π_θ | `pi_chosen_logp − pi_rejected_logp`; should grow > 0 as alignment improves |
| `train/kl_from_ref` | Mean KL divergence from the reference model | `mean(log π_θ(y) − log π_ref(y))` — measures how far π_θ has drifted from π_ref |
| `train/spin_lambda` | λ used this iteration | Read from training logs; controls the regularisation strength in SPIN loss |
| `train/learning_rate` | Current LR from the scheduler | Passed from HuggingFace Trainer logs |
| `train/grad_global_norm` | L2 norm of all gradients | `sqrt(sum(g.norm()² for g in grads))` — spike = gradient explosion |
| `train/weight_global_norm` | L2 norm of all trainable parameters | `sqrt(sum(w.norm()² for w in params))` — slow drift is normal |
| `train/weight_drift` | L2 norm of total weight change since iteration start | `sqrt(sum((w − w_init).norm()² for w in params))` — measures cumulative update magnitude |
| `train/throughput_sps` | Training samples per second | `per_device_train_batch_size / step_elapsed_seconds` |
| `train/perplexity` | Model perplexity (more interpretable than raw loss) | `exp(min(loss, 20))` — capped at `exp(20)` to avoid overflow |
| `train/alignment_accuracy` | Binary flag: 1.0 when margin_mean > 0 | `float(margin_mean > 0)` — 1.0 means the model already prefers human responses this step |

#### `system/*` — logged on every `on_step_end` call

| Metric | Description | Calculation |
|---|---|---|
| `system/gpu_alloc_mb` | GPU memory actively holding tensor data | `torch.cuda.memory_allocated() / 1024²` |
| `system/gpu_reserved_mb` | GPU memory reserved by the caching allocator (may exceed alloc) | `torch.cuda.memory_reserved() / 1024²` |
| `system/gpu_util_pct` | GPU compute utilisation percentage | Via `pynvml.nvmlDeviceGetUtilizationRates()`, fallback to `torch.cuda.utilization()` |
| `system/cpu_rss_mb` | Process RSS in CPU RAM | `psutil.Process(os.getpid()).memory_info().rss / 1024²` |

#### `gradients/*` and `weight_delta/*` — logged every `parameter_log_interval` steps (per layer)

| Metric | Description | Calculation |
|---|---|---|
| `gradients/norm/<layer>` | L2 norm of the gradient for this parameter | `param.grad.norm()` |
| `gradients/absmax/<layer>` | Max absolute gradient value for this parameter | `param.grad.abs().max()` |
| `gradients/<layer>` (histogram) | Full gradient histogram for this parameter | `add_histogram()` — enabled by `cfg.log_gradient_histograms` |
| `weight_delta/norm/<layer>` | L2 norm of weight change from iteration start | `(w − w_init).norm()` |
| `weight_delta/mean/<layer>` | Mean weight change from iteration start | `(w − w_init).mean()` |
| `layer_health/grad_weight_ratio/<layer>` | Gradient-to-weight ratio | `grad.norm() / (weight.norm() + 1e-8)` — detects vanishing/exploding gradients relative to parameter scale |

#### `epoch/*` — logged on every `on_epoch_end` call

| Metric | Description | Calculation |
|---|---|---|
| `epoch/loss_mean` | Average loss over the epoch | Mean of all step losses accumulated during the epoch |
| `epoch/loss_min` | Best (lowest) loss in the epoch | `min(epoch_losses)` |
| `epoch/loss_max` | Worst (highest) loss in the epoch | `max(epoch_losses)` |
| `epoch/loss_final` | Last step loss before epoch end | `epoch_losses[-1]` |
| `epoch/loss_distribution` | Histogram of all step losses during the epoch | `add_histogram()` |
| `epoch/margin_mean` | Average SPIN margin over the epoch | Mean of accumulated margin values |
| `epoch/margin_final` | Margin at the last step of the epoch | `epoch_margins[-1]` |
| `epoch/win_rate_mean` | Average win rate over the epoch | Mean of accumulated win_rate values |
| `epoch/win_rate_final` | Win rate at the last step of the epoch | `epoch_win_rates[-1]` |
| `epoch/logp_gap_mean` | Average logp gap over the epoch | Mean of `(pi_chosen_logp − pi_rejected_logp)` per step |
| `epoch/logp_gap_final` | LogP gap at the last step of the epoch | `epoch_gaps[-1]` |
| `train/alignment_pr_curve` | PR curve: precision-recall for alignment | Label=1 when `margin_mean > 0`; score=`sigmoid(margin_mean)` — enabled by `cfg.log_pr_curves` |

#### `iteration_summary/*` — logged once in `on_train_end`

| Metric | Description | Calculation |
|---|---|---|
| `iteration_summary/final_loss` | Loss at the last logged training step | From last `log_history` entry containing a loss value |
| `iteration_summary/final_margin_mean` | Margin at the last logged step | From last `log_history` entry |
| `iteration_summary/final_pi_chosen_logp` | Chosen log-prob at the last step | From last `log_history` entry |
| `iteration_summary/final_pi_rejected_logp` | Rejected log-prob at the last step | From last `log_history` entry |
| `iteration_summary/final_win_rate` | Win rate at the last step | From last `log_history` entry |
| `iteration_summary/final_kl_from_ref` | KL from ref at the last step | From last `log_history` entry |
| `iteration_summary/final_logp_gap` | Final logp gap | `pi_chosen_logp − pi_rejected_logp` from last `log_history` entry |
| `iteration_summary/total_weight_drift` | Total weight change over the full iteration | `sqrt(sum((w_final − w_init).norm()² for all trainable params))` |

#### `embeddings/*` — logged in `on_train_begin` and `on_train_end` (PROJECTOR tab)

Token embedding snapshots are written at iteration start (`step=0`) and iteration end (`step=global_step`). Enabled by `cfg.log_embedding_projector`. Labels are decoded token strings when a tokenizer is provided; otherwise plain token-ID strings.

#### `hparam/*` — logged once in `on_train_end` (HParams plugin)

Links hyperparameters to final metrics for cross-run comparison via TensorBoard's Parallel Coordinates and Scatter Plot views.

| Hyperparameter | Value |
|---|---|
| `learning_rate` | `args.learning_rate` |
| `batch_size` | `args.per_device_train_batch_size` |
| `num_epochs` | `args.num_train_epochs` |
| `spin_iteration` | Iteration index |
| `spin_lambda` | λ value from training logs |

Final metrics linked: `hparam/final_loss`, `hparam/final_margin`, `hparam/final_win_rate`, `hparam/final_logp_gap`, `hparam/final_kl_from_ref`.

---

## 2. SPINIterationSummaryCallback

**File:** [spin_iteration_summary_callback.py](spin_iteration_summary_callback.py)

Cross-iteration TensorBoard logger. Created **once** before the outer SPIN loop and reused across all iterations. Uses the SPIN iteration index as the x-axis, so every chart shows how the model improves from iteration 0 → 1 → 2 → ...

Call `set_iteration(i)` before each `trainer.train()` and `close()` after the loop ends.

### Constructor

```python
SPINIterationSummaryCallback(log_dir, cfg=None)
```

| Parameter | Description |
|---|---|
| `log_dir` | Directory for the global (cross-iteration) TensorBoard writer |
| `cfg` | `SPINConfig` — gates optional features like model graph logging |

---

### Metrics Written

All metrics use the SPIN iteration index as the x-axis (not the training step).

#### `spin_progress/*` — logged once per iteration in `on_train_end`

**Loss:**

| Metric | Description | Calculation |
|---|---|---|
| `spin_progress/final_loss` | Loss at the last step of the iteration | `losses[-1]` |
| `spin_progress/mean_loss` | Average loss over the full iteration | `mean(all step losses)` |
| `spin_progress/min_loss` | Best (lowest) loss seen in the iteration | `min(all step losses)` |

**Alignment signal:**

| Metric | Description | Calculation |
|---|---|---|
| `spin_progress/final_margin` | SPIN margin at the last step | `margins[-1]` — higher = better alignment |
| `spin_progress/mean_margin` | Average margin over the iteration | `mean(all step margins)` |
| `spin_progress/final_logp_gap` | Log-prob gap at the last step | `chosen_lps[-1] − rejected_lps[-1]` — should grow across iterations |
| `spin_progress/mean_logp_gap` | Average log-prob gap over the iteration | `mean(chosen_lp − rejected_lp)` per step |
| `spin_progress/final_pi_chosen_logp` | How likely the model generates human responses at end of iteration | `chosen_lps[-1]` |
| `spin_progress/final_pi_rejected_logp` | How likely the model generates synthetic responses at end of iteration | `rejected_lps[-1]` |
| `spin_progress/final_win_rate` | Win rate at the last step | `win_rates[-1]` |
| `spin_progress/mean_win_rate` | Average win rate over the iteration | `mean(all step win rates)` |
| `spin_progress/final_kl_from_ref` | KL from reference model at the last step | `kl_vals[-1]` |
| `spin_progress/mean_kl_from_ref` | Average KL from reference over the iteration | `mean(all step KL values)` |

**Model drift:**

| Metric | Description | Calculation |
|---|---|---|
| `spin_progress/weight_drift_from_iter_start` | Weight change within this iteration | `sqrt(sum((w_end − w_iter_start).norm()²))` for all trainable params |
| `spin_progress/weight_drift_from_base_model` | Cumulative drift from the original checkpoint | `sqrt(sum((w_end − w_base).norm()²))` for all trainable params — snapshot taken at iteration 0 |
| `spin_progress/cosine_sim_to_iter_start` | Directional similarity: 1.0 = no change | `cosine_similarity(flat_start_weights, flat_end_weights)` — lower = larger directional shift |

**Training stats:**

| Metric | Description | Calculation |
|---|---|---|
| `spin_progress/total_steps` | Optimizer steps taken in this iteration | `state.global_step` |
| `spin_progress/final_lr` | Learning rate at the end of this iteration | Last `learning_rate` value from training logs |
| `spin_progress/spin_lambda` | λ value used this iteration | From training logs |
| `spin_progress/dataset_size` | Number of examples in the training dataset | Passed via `set_iteration()` |

#### `spin_iterations/*` — text cards

| Tag | Content | When |
|---|---|---|
| `spin_iterations/log` | "Iteration N started — LR=..., epochs=..., batch=..." | `on_train_begin` per iteration |
| `spin_iterations/summary` | Markdown summary: loss, margin, win rate, logp gap, KL, λ, steps, dataset size | `on_train_end` per iteration |

#### `profiler/*` — text card (written by TorchProfilerCallback if `tb_writer` is provided)

| Tag | Content |
|---|---|
| `profiler/key_averages_iter_N` | Top-N operator table sorted by CUDA time (or CPU time for CPU-only runs) |

---

## 3. TensorBoardParameterStatsCallback

**File:** [tensorboard_parameter_stats_callback.py](tensorboard_parameter_stats_callback.py)

Snapshots parameter statistics at the start and end of each SPIN iteration and writes them to a dedicated `param_stats/<run_name>` TensorBoard subdirectory. Tracks per-layer statistics and how far each layer has moved from its initial weights.

### Constructor

```python
TensorBoardParameterStatsCallback(cfg, run_name)
```

| Parameter | Description |
|---|---|
| `cfg` | `SPINConfig` controlling which stats are logged and how many tensors to track |
| `run_name` | Sub-directory name under `cfg.tensorboard_dir/param_stats/` |

---

### Metrics Written

All per-layer tags use `/` as the separator (e.g., `model.layers.0.self_attn.q_proj.weight` → `model/layers/0/self_attn/q_proj/weight`).

#### `parameters/*` — logged at step 0 (`on_train_begin`) and every `parameter_log_interval` steps, and at `on_train_end`

| Metric | Description | Calculation | Config gate |
|---|---|---|---|
| `parameters/<layer>` (histogram) | Full weight distribution for this parameter | `add_histogram(param)` | `cfg.log_parameter_histograms` |
| `parameters/mean/<layer>` | Mean of the weight tensor | `param.mean()` | `cfg.log_parameter_scalars` |
| `parameters/std/<layer>` | Std-dev of the weight tensor | `param.std()` | `cfg.log_parameter_scalars` |
| `parameters/norm/<layer>` | L2 norm of the weight tensor | `param.norm()` | `cfg.log_parameter_scalars` |
| `parameters/absmax/<layer>` | Max absolute value of the weight tensor | `param.abs().max()` | `cfg.log_parameter_scalars` |

#### `parameter_delta/*` — logged every `parameter_log_interval` steps (when `initial_params` baseline exists)

| Metric | Description | Calculation |
|---|---|---|
| `parameter_delta/norm/<layer>` | L2 norm of weight change from iteration start | `(w − w_init).norm()` |
| `parameter_delta/mean/<layer>` | Mean weight change from iteration start | `(w − w_init).mean()` |
| `parameter_delta/std/<layer>` | Std-dev of weight change from iteration start | `(w − w_init).std()` |

#### `gradients/*` — logged every `parameter_log_interval` steps (when gradients are available)

| Metric | Description | Calculation | Config gate |
|---|---|---|---|
| `gradients/<layer>` (histogram) | Full gradient distribution for this parameter | `add_histogram(grad)` | `cfg.log_gradient_histograms` |
| `gradients/norm/<layer>` | L2 norm of gradient | `grad.norm()` | `cfg.log_gradient_histograms` |
| `gradients/absmax/<layer>` | Max absolute gradient value | `grad.abs().max()` | `cfg.log_gradient_histograms` |

**Logging schedule:** Steps `1` (first optimizer step) and every `cfg.parameter_log_interval` steps thereafter. `on_train_end` always writes a final snapshot, guaranteeing at least 2 histogram time points so TensorBoard Distributions can render the percentile band chart.

---

## 4. MemoryProbeCallback

**File:** [memory_probe_callback.py](memory_probe_callback.py)

Lightweight callback that logs CPU RSS and GPU memory to the Python logger and optionally to TensorBoard at configurable intervals. Designed to pinpoint exactly which step causes a memory spike.

### Constructor

```python
MemoryProbeCallback(writer=None, log_every_n_steps=50)
```

| Parameter | Description |
|---|---|
| `writer` | Optional `SummaryWriter`; metrics are TensorBoard-logged only when provided |
| `log_every_n_steps` | Throttle interval. The first 3 steps are always logged regardless of this value to capture warmup spikes |

---

### Metrics Written

| Metric | Description | Calculation |
|---|---|---|
| `system/gpu_alloc_mb` | GPU memory actively holding tensor data | `torch.cuda.memory_allocated() / 1024²` |
| `system/gpu_reserved_mb` | GPU memory reserved by the caching allocator | `torch.cuda.memory_reserved() / 1024²` |
| `system/cpu_rss_mb` | Process RSS (resident set size) in CPU RAM | `psutil.Process(os.getpid()).memory_info().rss / 1024²` |

### Logging schedule

| Event | When |
|---|---|
| `on_train_begin` | Always — baseline snapshot before the first step |
| `on_step_end` (steps 0–2) | Always — captures warmup allocation spikes |
| `on_step_end` (step ≥ 3) | When `step % log_every_n_steps == 0` |
| `on_train_end` | Always — final snapshot after all training |

All events are also written to the Python logger via `log_memory(tag)` regardless of whether a `SummaryWriter` is provided.

---

## 5. TorchProfilerCallback

**File:** [torch_profiler_callback.py](torch_profiler_callback.py)

Runs the PyTorch profiler during training and exports rich diagnostics. A new instance is created per SPIN iteration so profiler output from each iteration is isolated.

### Constructor

```python
TorchProfilerCallback(cfg, spin_iteration=0, tb_writer=None)
```

| Parameter | Description |
|---|---|
| `cfg` | `SPINConfig` controlling profiler schedule, activity flags, and output options |
| `spin_iteration` | SPIN iteration index — used to name the output directory |
| `tb_writer` | Optional `SummaryWriter` (pass `summary_cb.writer` from `SPINIterationSummaryCallback`) to receive the key-averages text card |

Enabled only when `cfg.enable_profiler = True`.

---

### Outputs Written

All outputs go to `cfg.profile_dir/run_iter_N_step_S/`.

| File | Description | How to View |
|---|---|---|
| `*.pt.trace.json` | Operator timeline trace | TensorBoard "Trace" tab or `chrome://tracing` |
| `stacks_cuda.txt` | CUDA flamegraph stacks | Drag into [speedscope.app](https://speedscope.app) |
| `stacks_cpu.txt` | CPU flamegraph stacks (Python overhead, host-side bottlenecks) | Drag into [speedscope.app](https://speedscope.app) |
| `trace_fallback.pt.trace.json` | Fallback trace when training is too short for the scheduled trace to fire | TensorBoard "Trace" tab |

### TensorBoard Text Card

| Tag | Content |
|---|---|
| `profiler/key_averages_iter_N` | Top-`cfg.profile_log_top_n_ops` operators sorted by CUDA time (or CPU time for CPU-only runs), optionally grouped by input shape and stack depth |

### Profiler Schedule (`SPINConfig` fields)

| Config field | Description |
|---|---|
| `profile_schedule_wait` | Steps to wait before starting profiling |
| `profile_schedule_warmup` | Steps to warm up the profiler (events captured but not exported) |
| `profile_schedule_active` | Steps to actively capture and export |
| `profile_schedule_repeat` | Number of times to repeat wait→warmup→active (0 = repeat until end) |
| `profile_cpu` | Capture CPU events |
| `profile_cuda` | Capture CUDA kernel events |
| `profile_memory` | Profile memory allocation/deallocation |
| `profile_with_stack` | Include Python call stacks (required for flamegraph export) |
| `profile_with_flops` | Estimate FLOPs per operator |
| `profile_modules` | Group ops by `nn.Module` name instead of raw ATen ops |
| `profile_record_shapes` | Record input tensor shapes (groups table rows by shape) |
| `profile_export_stacks` | Export CPU and CUDA flamegraph stack files |
| `profile_log_top_n_ops` | Number of top operators to log in the key-averages table (0 = skip) |

### Lifecycle

| Hook | Action |
|---|---|
| `on_train_begin` | Starts the profiler with the configured schedule and activities |
| `on_step_end` | Advances the profiler schedule by one step (`prof.step()`) |
| `on_train_end` | Stops the profiler, exports stacks, logs key-averages, lists all written files |

---

## Callback Co-usage

The callbacks are complementary and designed to be used together:

```python
summary_cb = SPINIterationSummaryCallback(log_dir="tb_global", cfg=cfg)

for iteration in range(cfg.num_spin_iterations):
    summary_cb.set_iteration(iteration, spin_lambda=cfg.spin_lambda)

    tb_cb = TensorBoardCallbackExtended(
        log_dir=f"tb_iter_{iteration}", cfg=cfg, iteration=iteration)
    param_cb = TensorBoardParameterStatsCallback(cfg=cfg, run_name=f"iter_{iteration}")
    mem_cb   = MemoryProbeCallback(writer=tb_cb.writer, log_every_n_steps=50)
    prof_cb  = TorchProfilerCallback(cfg=cfg, spin_iteration=iteration, tb_writer=summary_cb.writer)

    trainer = SPINTrainer(
        callbacks=[summary_cb, tb_cb, param_cb, mem_cb, prof_cb], ...)
    trainer.train()

summary_cb.close()
```

| Callback | Scope | x-axis |
|---|---|---|
| `TensorBoardCallbackExtended` | Per-iteration | Training step |
| `TensorBoardParameterStatsCallback` | Per-iteration | Training step |
| `MemoryProbeCallback` | Per-iteration | Training step |
| `TorchProfilerCallback` | Per-iteration | N/A (file export) |
| `SPINIterationSummaryCallback` | Global (all iterations) | SPIN iteration index |
