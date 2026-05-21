from dataclasses import dataclass
from typing import Optional


@dataclass
class SPINConfig:

    # ── Device ───────────────────────────────────────────────────────────────

    # Target compute device for model training and inference.
    # Values: "cuda" (GPU, required for practical training) | "cpu" (debug only, ~100x slower)
    device: str = "cuda"

    # ── Model ────────────────────────────────────────────────────────────────

    # HuggingFace Hub model ID (e.g. "meta-llama/Llama-3.2-1B") or an absolute local
    # path to a directory containing config.json + model weights. This is both the
    # starting checkpoint for iteration 0 and the reference model for SPIN iteration 0.
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Path to a tokenizer directory or Hub ID. If None, the tokenizer is loaded from
    # model_name_or_path. Useful when the tokenizer lives in a different repo than the weights.
    tokenizer_name_or_path: Optional[str] = None

    # Whether to allow the Hub to execute custom modelling code bundled with the model.
    # Required for models like Phi-3, Falcon, and other non-standard architectures.
    # Values: False (safe default) | True (only for trusted model repos)
    trust_remote_code: bool = False

    # Floating-point precision used when loading and training the model.
    # "bfloat16" — recommended; same dynamic range as float32, half the memory, stable training.
    # "float16"  — half memory but narrower range; may cause NaN on large models without careful LR.
    # "float32"  — full precision; use only when debugging numerical issues, 2x the memory.
    torch_dtype: str = "bfloat16"

    # Attention kernel implementation.
    # None                — default PyTorch eager attention (compatible with all hardware).
    # "flash_attention_2" — requires flash-attn package + Ampere+ GPU; 2-4x faster, uses less memory.
    # "sdpa"              — PyTorch scaled_dot_product_attention; free on PyTorch 2.0+, good default.
    attn_implementation: Optional[str] = "sdpa"

    # ── Data ─────────────────────────────────────────────────────────────────

    # HuggingFace Hub dataset name. If data_path is also set, data_path takes priority.
    # Must contain prompt/response pairs (or a conversations field).
    dataset_name: Optional[str] = "HuggingFaceH4/ultrachat_200k"

    # Subset / config name passed as the second argument to load_dataset().
    # Required for datasets that have multiple configs (e.g. "en" or "default").
    dataset_config_name: Optional[str] = None

    # Dataset split to load for training (e.g. "train", "train_sft", "train[:10%]").
    train_split: str = "train_sft"

    # Path to a local data file. Supported formats: .jsonl, .json, .parquet.
    # When set, overrides dataset_name. Each record must contain prompt and response fields.
    data_path: Optional[str] = None

    # Name of the column in the dataset that contains the user prompt / instruction.
    prompt_field: str = "prompt"

    # Name of the column that contains the ground-truth human response (the "chosen" side).
    response_field: str = "response"

    # ── Formatting ───────────────────────────────────────────────────────────

    # Controls how the prompt is wrapped before tokenisation.
    # "auto"                — use the tokenizer's built-in chat_template if present, else fall back to instruction_response mode.
    # "plain"               — pass the raw prompt string with no wrapping; suitable for base models.
    # "instruction_response"— manually prepend instruction_prefix and append response_prefix.
    chat_template_mode: str = "auto"

    # String prepended to the prompt when chat_template_mode="instruction_response".
    # Change to match the format the model was pre-trained with.
    instruction_prefix: str = "Instruction:\n"

    # String inserted between the prompt and the model response in instruction_response mode.
    response_prefix: str = "\n\nResponse:\n"

    # When True, appends the tokenizer's EOS token to every response before tokenisation.
    # Critical for teaching the model when to stop generating; always keep True.
    add_eos_to_response: bool = True

    # ── Sequence lengths ─────────────────────────────────────────────────────

    # Maximum number of tokens kept from the prompt before synthetic generation.
    # Longer prompts are truncated from the left (see truncation_side).
    # Reducing this is the fastest way to lower GPU memory: activation memory ∝ seq_len².
    # Recommended range: 128–1024. For an 8 GB GPU, keep ≤ 512.
    max_prompt_length: int = 256

    # Hard cap on total tokens (prompt + response) fed into the model during training.
    # Sequences longer than this are truncated. Activation memory scales as O(seq_len²)
    # for standard attention and O(seq_len) for Flash Attention.
    # Recommended range for 8 GB GPU: 512–1024.
    max_length: int = 512

    # Which end of an overlong sequence to truncate.
    # "left"  — drops tokens from the beginning of the prompt (preserves the question tail).
    # "right" — drops tokens from the end of the response (model never sees the full answer).
    truncation_side: str = "left"

    # ── Generation (synthetic response production) ───────────────────────────

    # Number of prompts decoded in a single GPU batch during synthetic generation.
    # KV-cache peak = batch × (max_prompt + max_new_tokens) × layers × KV-heads × head_dim × 2B.
    # SmolLM2-135M: 2 × 3 KV-heads × 64 head_dim × 30 layers × 2B = 22.5 KB per token.
    # batch=256, 512 tokens: KV cache ≈ 2.95 GB + model 0.27 GB = ~3.2 GB — fits on 8 GB.
    # batch=128: ~1.7 GB — overly conservative; 256 is safe and 2× faster generation.
    generation_batch_size: int = 256

    # Maximum number of new tokens the model may produce per response.
    # Longer responses create richer training signal but increase generation time linearly.
    # Range: 64–1024. Keep in mind: generation_max_new_tokens ≤ max_length − prompt_length.
    generation_max_new_tokens: int = 256

    # Whether to use stochastic sampling during generation.
    # True  — sample from the distribution; produces diverse, varied responses.
    # False — greedy decoding; deterministic but often repetitive and low-diversity.
    generation_do_sample: bool = True

    # Softmax temperature applied to logits before sampling.
    # Range: 0.1–2.0. Lower = more peaked / conservative; higher = more uniform / creative.
    # Only active when generation_do_sample=True. Typical: 0.7–1.0.
    generation_temperature: float = 0.9

    # Top-p (nucleus) sampling threshold: the model samples from the smallest set of tokens
    # whose cumulative probability exceeds this value.
    # Range: 0.0–1.0. Lower = more focused; 0.95 is a robust default.
    generation_top_p: float = 0.95

    # Top-k sampling: restrict sampling to the k most probable tokens.
    # 0 = disabled (use top_p only). Typical values when enabled: 20–100.
    generation_top_k: int = 0

    # Number of beams for beam search. 1 = pure sampling (stochastic); >1 = beam search
    # (deterministic, higher quality but much slower and incompatible with do_sample=True).
    generation_num_beams: int = 1

    # Repetition penalty applied to tokens already present in the context.
    # 1.0 = no penalty. >1.0 = actively discourages repetition (try 1.1–1.3 for chatty models).
    generation_repetition_penalty: float = 1.0

    # Whether to cache key/value states across decoding steps (KV cache).
    # True = faster generation (recommended). False = lower peak memory during generation.
    generation_use_cache: bool = True

    # Number of rows scored in a single forward pass during compute_ref_logprobs().
    # Reduce if ref-logprob scoring causes OOM (each batch holds two padded sequences).
    # logits tensor = batch × seq_len × vocab_size × 2B.
    # SmolLM2-135M (vocab=49152, max_length=512): batch=32 → 32×512×49152×2B ≈ 1.5 GB.
    # Safe on 8 GB; 2× faster than batch=16.
    ref_logprob_batch_size: int = 32

    # ── SPIN training loop ───────────────────────────────────────────────────

    # Total number of SPIN outer iterations. Each iteration:
    #   1. Uses the current model to generate synthetic responses (the "opponent").
    #   2. Trains a new model to prefer human responses over those synthetic ones.
    # More iterations = more self-improvement cycles. Diminishing returns after 3–5.
    # Range: 1–10. Typical: 3–5.
    num_iterations: int = 5

    # Number of full passes over the synthetic dataset inside a single SPIN iteration.
    # More epochs = stronger fitting to current synthetic data, but risks overfitting.
    # Range: 1–5. Typical: 1–3.
    num_epochs_per_iteration: int = 2

    # Hard cap on the total number of records read from the dataset at load time.
    # Applied in load_base_dataset_fixed() before any per-iteration sampling.
    # Use this to bound memory and startup time when the source dataset is very large
    # (e.g. ultrachat_200k has ~200 k rows; setting this to 50 000 loads only the first 50 k).
    # 0 = load the full dataset split.
    # Range: 0 (unlimited) or any positive integer ≤ dataset size.
    max_data_load: int = 207865

    # Number of dataset rows processed as one atomic checkpoint unit during
    # synthetic generation and ref-logprob scoring. Each batch is saved to
    # iter_{i}_batch_{k:06d}.jsonl before the next batch starts, so a crash
    # loses at most one batch worth of GPU work.
    # Smaller → more frequent saves, lower restart cost.
    # Larger  → fewer file writes, but more work lost per crash.
    # Range: 50–10000. Start at 200 and tune for your restart tolerance.
    data_batch_size: int = 65536

    # λ (lambda) applied in all iterations except the last.
    # Scales the SPIN margin: margin = λ × [(π_θ(chosen) − π_ref(chosen)) − (π_θ(rejected) − π_ref(rejected))].
    # Larger λ = stronger gradient signal, but too large can destabilise training.
    # NOTE: log-probs are per-token averages (~-0.5 to -2.0), so λ must be larger than
    # the raw-sum regime (~-50 to -500) to produce the same effective margin scale.
    # Range: 1–50 with per-token normalization. Typical: 10.
    lambda_initial: float = 10.0

    # λ used exclusively in the final SPIN iteration (if final_iteration_lambda_only=True).
    # A much larger value here applies a strong final alignment push.
    # Range: 10–100 with per-token normalization.
    lambda_final_iteration: Optional[float] = 50.0

    # When True, lambda_final_iteration replaces lambda_initial only for the very last
    # iteration; all earlier iterations still use lambda_initial.
    # When False, lambda_final_iteration is ignored and lambda_initial is used throughout.
    final_iteration_lambda_only: bool = True

    # Loss function applied to the SPIN margin. All variants are minimised when margin > 0.
    # "logistic"    — softplus(−margin): smooth, never saturates, recommended.
    # "hinge"       — relu(1 − margin): zero loss once margin > 1; hard boundary.
    # "correlation" — (1 − margin): linear penalty; constant gradient, easiest to tune.
    # "exponential" — exp(−margin): very aggressive for negative margins; can cause instability.
    loss_type: str = "logistic"

    # ── Training hyperparameters ─────────────────────────────────────────────

    # Root directory where all run outputs are written (config snapshot, logs, checkpoints).
    output_dir: str = "./spin_outputs"

    # Number of training examples processed per GPU per optimizer step.
    # This is the primary knob for GPU memory: activation memory scales linearly with it.
    # For an 8 GB GPU with a ~1B parameter model: use 1.
    # For a 24 GB GPU: try 4–8.
    # Range: 1–32 (GPU-memory dependent).
    per_device_train_batch_size: int = 4

    # Gradients are accumulated over this many forward passes before one optimizer step.
    # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps.
    # Increase this to compensate when you lower per_device_train_batch_size to fit in memory.
    # Range: 1–512. Typical: 32–128.
    gradient_accumulation_steps: int = 32

    # Peak learning rate used during early SPIN iterations (iterations < late_lr_start_iteration).
    # Very small values prevent catastrophic forgetting of pre-trained knowledge.
    # Range: 1e-7–5e-6. Typical: 5e-7 for 7B models; ~1e-6 for 135M-scale models.
    learning_rate: float = 1e-6

    # Learning rate used from late_lr_start_iteration onward.
    # Smaller than learning_rate to allow fine-grained alignment in later iterations.
    # Range: 1e-8–1e-6. Typical: 1e-7.
    learning_rate_late: float = 2e-7

    # SPIN iteration index (0-based) at which the LR switches from learning_rate to learning_rate_late.
    # E.g. 2 means iterations 0,1 use learning_rate and iterations 2+ use learning_rate_late.
    late_lr_start_iteration: int = 2

    # L2 regularisation coefficient applied to weight matrices (not biases or layer norms).
    # 0.0 is standard for supervised fine-tuning. Small values (1e-4) can help generalisation.
    # Range: 0.0–0.1.
    weight_decay: float = 0.0

    # Number of linear LR warmup steps at the beginning of each iteration.
    # Must be small relative to total optimizer steps per SPIN batch.
    # With per_device=16, GA=32, data_batch=16384, epochs=2:
    #   total optimizer steps per batch = (16384/16/32)*2 = 64
    # So warmup_steps=5 → ~8% warmup, which is correct.
    # (Large values like 50 would make 78% of training run be warmup — broken.)
    warmup_steps: int = 5

    # Learning rate scheduler shape after warmup.
    # "cosine"  — smooth decay to 0; best for fine-tuning.
    # "linear"  — linear decay to 0.
    # "constant"— no decay; rarely used for fine-tuning.
    lr_scheduler_type: str = "cosine"

    # Optimizer algorithm.
    # "rmsprop" — 1 state tensor per parameter (running mean of squared gradients); lower GPU memory.
    # "adamw"   — 2 state tensors per parameter (first + second moment); better convergence, more memory.
    # For an 8 GB GPU, prefer "rmsprop" to save ~2 GB of optimizer state.
    optimizer: str = "adamw"

    # Maximum L2 norm of the gradient vector before clipping is applied.
    # Prevents a single bad batch from causing a catastrophic parameter update.
    # Range: 0.1–10.0. Typical: 1.0.
    max_grad_norm: float = 1.0

    # Frequency (in optimizer steps) at which training metrics are written to the log.
    # Lower = more granular progress but slightly more I/O overhead.
    # With per_device=16, GA=32, data_batch=16384, epochs=2: total steps = 64 per SPIN batch.
    # logging_steps=10 → ~6 log events per batch; logging_steps=50 → only 1 event (too sparse).
    # Range: 1–500. Typical: 10–50.
    logging_steps: int = 10

    # When to save checkpoints.
    # "epoch" — save once per training epoch (default, safe).
    # "steps" — save every save_steps steps (requires setting save_steps in TrainingArguments).
    # "no"    — never save intermediate checkpoints.
    save_strategy: str = "epoch"

    # Maximum number of checkpoints to retain on disk. Older checkpoints are deleted automatically.
    # Set to 1 so only the latest mid-iteration HF Trainer checkpoint is kept; the merged
    # model saved to iter_dir after training supersedes all of them.
    save_total_limit: int = 1

    # Enable bfloat16 mixed-precision training. Halves GPU memory for activations and
    # intermediate tensors. Requires Ampere+ GPU (RTX 30xx, A100, H100).
    # Cannot be True at the same time as fp16.
    bf16: bool = True

    # Enable float16 mixed-precision training. Alternative to bf16 for older GPUs (Volta, Turing).
    # Less numerically stable than bf16; may require loss scaling to prevent NaN.
    # Cannot be True at the same time as bf16.
    fp16: bool = False

    # When True, intermediate activations are discarded during the forward pass and recomputed
    # on demand during the backward pass. Reduces activation memory by ~10x at the cost of
    # ~33% more compute. Essential for fitting large models on small GPUs — enable if OOM.
    gradient_checkpointing: bool = False

    # Number of subprocess workers used by the DataLoader to prefetch batches.
    # 0 — all data loading happens in the main process (required on Windows to avoid deadlocks).
    # >0 — parallel prefetching (faster on Linux); each worker forks the entire process.
    # Range: 0–8. Use 0 on Windows; 2–4 on Linux.
    dataloader_num_workers: int = 0

    # Pin DataLoader output tensors into page-locked (pinned) CPU memory before
    # transferring to the GPU. Enables async DMA so the CPU→GPU transfer overlaps
    # with compute. Almost always a win when training on GPU; set False only if
    # you are CPU-memory constrained (pinned memory cannot be swapped to disk).
    dataloader_pin_memory: bool = True

    # Must remain False. The HuggingFace Trainer strips columns not in the model's forward()
    # signature when True; our custom keys (ref_chosen_logp, ref_rejected_logp) would be dropped.
    remove_unused_columns: bool = False

    # Experiment tracking backend.
    # "tensorboard" — logs to local TensorBoard files (no account required).
    # "none"        — disables all external logging.
    report_to: str = "tensorboard"

    # Path to a DeepSpeed JSON configuration file for ZeRO-stage memory offloading.
    # None = DeepSpeed disabled. ZeRO-2/3 can enable training models larger than GPU memory
    # by offloading optimizer states (ZeRO-2) or parameters (ZeRO-3) to CPU RAM.
    deepspeed: Optional[str] = None

    # ── torch.compile ────────────────────────────────────────────────────────

    # Apply torch.compile() to the trainable model before each iteration's training.
    # Requires PyTorch >= 2.0. Typically yields 10–30% throughput gains on Ampere+ GPUs
    # via kernel fusion and graph optimisation. Disable if you see compilation errors
    # (common with certain LoRA target module combinations or older triton versions).
    compile_model: bool = True

    # Backend passed to torch.compile().
    # "inductor"   — default; best GPU throughput via Triton kernel fusion (requires triton pkg).
    # "aot_eager"  — AOT Autograd with eager execution; no Triton needed; useful for debugging.
    # "eager"      — disables actual compilation; effectively a no-op useful for A/B testing.
    compile_backend: str = "inductor"

    # Trade-off between compile latency and runtime speed.
    # "default"                    — balanced; good for most use cases.
    # "reduce-overhead"            — minimises Python/CUDA kernel launch overhead; best for small batches.
    # "max-autotune"               — exhaustive Triton kernel search + CUDAGraphs; highest throughput for
    #                                single-forward-per-step workloads (e.g. ref model scoring).
    # "max-autotune-no-cudagraphs" — same kernel search but without CUDAGraph memory reuse. CUDAGraph
    #                                capture requires a CPU-side C++ launcher compiled with OpenMP (omp.h),
    #                                which is not available in this MSVC setup on Windows. Use this mode
    #                                to get Triton kernel tuning without the C++ compilation step.
    compile_mode: str = "default"

    # compile_dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
    # to generate a kernel that is as dynamic as possible to avoid recompilations when
    # sizes change.  This may not always work as some operations/optimizations will
    # force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
    # When this is False, we will NEVER generate dynamic kernels, we will always specialize.
    # By default (None), we automatically detect if dynamism has occurred and compile a more
    # dynamic kernel upon recompile.
    compile_dynamic: bool = True

    # Require the entire forward graph to compile without breaks (fullgraph=True).
    # More performant when it succeeds, but raises if the model contains graph-break ops.
    # Try False first; switch to True only after confirming your model compiles cleanly.
    compile_fullgraph: bool = True

    # Also compile the frozen reference model used for log-prob scoring and synthetic generation.
    # Disabled: model.generate() uses a Python while-loop that always causes a graph break,
    # so compile_fullgraph=True fails silently (caught by maybe_compile_model's try/except)
    # and falls back to eager — paying max-autotune search time for zero runtime benefit.
    compile_ref_model: bool = False

    # ── LoRA / PEFT ──────────────────────────────────────────────────────────

    # Enable Low-Rank Adaptation (LoRA). Instead of updating all parameters, LoRA inserts
    # small trainable rank-r matrices alongside frozen weight matrices. This reduces the
    # number of trainable parameters from billions to millions, cutting GPU memory for
    # gradients and optimizer states by 10–100x. Highly recommended for small GPUs.
    use_lora: bool = True

    # LoRA rank: the inner dimension of the two low-rank matrices A (d×r) and B (r×k).
    # Higher rank = more expressive adapters = more parameters and memory.
    # Range: 4–128. Typical: 8 (memory-constrained) or 16–32 (standard).
    lora_r: int = 16

    # LoRA scaling factor α. The adapter output is scaled by α/r before being added to
    # the frozen weight output. Effective adapter learning rate ≈ lr × (α / r).
    # Common practice: set α = 2 × r for a scale of 2.0, or α = r for scale 1.0.
    # Range: 8–128.
    lora_alpha: int = 32

    # Dropout probability applied to the LoRA adapter outputs during training.
    # Acts as regularisation to prevent adapter overfitting on small datasets.
    # Range: 0.0–0.2. Typical: 0.05.
    lora_dropout: float = 0.05

    # Comma-separated list of nn.Linear layer name suffixes that receive LoRA adapters.
    # SmolLM2-135M-Instruct uses LLaMA-style attention: "q_proj,k_proj,v_proj,o_proj".
    # GPT-2 / distilgpt2 family uses: "c_attn,c_proj".
    # make_trainable() will auto-detect the correct names if these aren't found in the model.
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    # ── Misc ─────────────────────────────────────────────────────────────────

    # Master random seed passed to set_seed(); controls weight initialisation, data shuffling,
    # and sampling. Change to run with different randomness while keeping everything else fixed.
    seed: int = 42

    # Directory where per-iteration synthetic JSONL files are saved.
    # One file per iteration: iter_0.jsonl, iter_1.jsonl, …
    synthetic_cache_dir: str = "./spin_outputs/synthetic"

    # Directory where per-iteration trained model checkpoints are saved.
    # Structure: checkpoints_dir/iter_0/, checkpoints_dir/iter_1/, …
    checkpoints_dir: str = "./spin_outputs/checkpoints"

    # Root directory for all TensorBoard event files.
    # Subdirectories: global/, iter_N/, param_stats/spin_iter_N/, profile/
    # Point the TensorBoard server at this single directory:
    #   tensorboard --logdir ./spin_outputs/tensorboard
    tensorboard_dir: str = "./spin_outputs/tensorboard"

    # ── TensorBoard profiler ─────────────────────────────────────────────────

    # Whether to run the PyTorch profiler. When enabled, records operator-level GPU/CPU
    # timelines and memory traces viewable in TensorBoard under the "Trace" and "Memory" tabs.
    # Disable in production runs to avoid the ~5% overhead.
    enable_profiler: bool = False

    # Directory where profiler trace files (.pt.trace.json) are written.
    profile_dir: str = "./spin_outputs/tensorboard/profile"

    # Number of steps to skip at the start before the profiler begins collecting.
    # The profiler needs (wait + warmup + active) steps before any trace is written.
    # Set to 0 to start profiling immediately — required when training runs ≤ 5 steps.
    # Range: 0–10.
    profile_schedule_wait: int = 0

    # Number of steps the profiler runs in "warmup" mode: the profiler is active but
    # data is discarded. Ensures the profiler itself is warmed up before recording.
    # Set to 0 when training runs very few steps so the active phase is guaranteed to fire.
    # Range: 0–5.
    profile_schedule_warmup: int = 0

    # Number of steps during which profiler data is actually recorded and saved.
    # Keep small (1–5) — a single active step produces several hundred MB of trace data.
    # Must be ≤ total training steps or the schedule will never fire (no trace written).
    profile_schedule_active: int = 1

    # How many full wait/warmup/active cycles to execute before stopping.
    # 1 = profile once near the beginning and stop; 0 = profile indefinitely.
    profile_schedule_repeat: int = 1

    # Record the input/output tensor shapes for each profiled operation.
    # Useful for spotting unexpected large intermediate tensors. Adds minor overhead.
    profile_record_shapes: bool = True

    # Record GPU memory allocation and deallocation events in the trace.
    # Essential for diagnosing OOM — shows exactly which op triggers the spike.
    profile_memory: bool = True

    # Capture the Python call stack at the point each operation is launched.
    # Allows you to trace an OOM or slow op directly back to the source line.
    # Adds noticeable overhead; disable if trace files become too large.
    profile_with_stack: bool = True

    # Estimate floating-point operation counts (FLOPs) for supported operations.
    # Useful for measuring hardware utilisation (MFU).
    profile_with_flops: bool = True

    # Profile at the nn.Module granularity rather than individual ATen operators.
    # Produces a coarser but more readable breakdown by layer name.
    profile_modules: bool = False

    # Include CPU-side activity (Python overhead, data loading, host-to-device copies) in the trace.
    profile_cpu: bool = True

    # Include CUDA kernel execution times in the trace.
    # Disable only if profiling a CPU-only run.
    profile_cuda: bool = True

    # How many top operators to print to the logger (and TensorBoard text card) after profiling.
    # Operators are sorted by CUDA time (or CPU time for CPU-only runs). Set to 0 to disable.
    # Range: 0–50. 20 is a good default — captures the critical path without flooding the log.
    profile_log_top_n_ops: int = 20

    # Export CPU and CUDA flamegraph stack files (.txt) alongside the TensorBoard trace.
    # Open the exported files at https://speedscope.app for interactive flamegraphs.
    # Requires profile_with_stack=True; no-op otherwise.
    profile_export_stacks: bool = True

    # ── TensorBoard parameter tracking ───────────────────────────────────────

    # Log per-parameter weight value histograms to TensorBoard.
    # Useful for detecting dead neurons (weight values collapsing to zero) or
    # exploding weights (distribution spreading extremely wide). High storage cost.
    log_parameter_histograms: bool = True

    # Log per-parameter gradient histograms to TensorBoard.
    # Useful for diagnosing vanishing gradients (histogram near zero) or
    # exploding gradients (histogram spread over large values).
    log_gradient_histograms: bool = True

    # Log scalar statistics (mean, std, L2 norm) for each parameter tensor.
    # Much lower storage overhead than full histograms; a good default to leave on.
    log_parameter_scalars: bool = True

    # Log parameter statistics every N optimizer steps.
    # Higher values reduce TensorBoard file size and logging overhead.
    # With 64 total optimizer steps per SPIN batch, 500 never fires — use 20 instead.
    # 20 → logs at steps 20, 40, 60 (~3× per batch); safe overhead.
    # Range: 10–500. Typical: 20–50.
    parameter_log_interval: int = 20

    # Maximum number of parameter tensors to log per step.
    # Prevents TensorBoard from becoming unresponsive when the model has thousands of layers.
    # Range: 10–1000. Reduce if TensorBoard is slow to load.
    parameter_log_max_tensors: int = 200

    # Log a summary of trainable vs total parameter counts when a model is prepared
    # for training. Disabled by default to keep logs quiet during normal runs.
    log_trainable_parameters: bool = True

    # ── TensorBoard visualization extras ─────────────────────────────────────

    # Log per-epoch Precision-Recall curves for alignment accuracy.
    # Labels = (margin > 0) per batch; scores = sigmoid(margin_mean).
    # Activates the PR CURVES tab in TensorBoard.
    log_pr_curves: bool = True

    # Log token embedding projections at the start and end of each SPIN iteration.
    # Activates the PROJECTOR tab — lets you visualise how token embeddings shift
    # across iterations using PCA / UMAP / t-SNE inside TensorBoard.
    log_embedding_projector: bool = True

    # Number of tokens (rows of the embedding matrix) to include in the projection.
    # Full vocab is often 32k–128k tokens which makes projection too slow in the browser.
    # 2048 covers the most common tokens and keeps the projector fast.
    # Range: 256–8192.
    embedding_projector_n_tokens: int = 2048

    # Attempt to trace and log the model's computation graph.
    # Activates the GRAPHS tab in TensorBoard.
    # Disabled by default: large transformer models are slow to trace and often produce
    # unreadable graphs. Enable only for small/debug models or architecture inspection.
    log_model_graph: bool = True

    # ── Evaluation ───────────────────────────────────────────────────────────────

    # Directory for per-iteration JSON eval results and the comparative summary.
    # Relative to the same root as output_dir.
    eval_output_dir: str = "./spin_outputs/eval_results"

    # TensorBoard log directory for evaluation metrics (comparison across iterations).
    eval_tensorboard_dir: str = "./spin_outputs/tensorboard/eval_compare"

    # Max examples per task during evaluation (None = full dataset).
    # Set to a small integer (e.g. 50) for a quick smoke test; leave None for real benchmarks.
    eval_limit: Optional[int] = None

    # Number of (context, continuation) rows per GPU forward pass during evaluation.
    # Logits per batch = batch × actual_seq_len × vocab_size × 2B.
    # Real eval sequences (ARC, TruthfulQA, Winogrande) average 100–400 tokens, not 2048.
    # SmolLM2-135M: batch=8 × 512 tokens × 49152 vocab × 2B ≈ 0.4 GB — safe on 8 GB.
    eval_batch_size: int = 8

    # Maximum total token length (context + continuation) fed to the model during evaluation.
    # Sequences longer than this are truncated from the left.
    eval_max_seq_len: int = 2048

    # Maximum new tokens generated per response in the GSM8k benchmark (generation task).
    eval_gsm8k_max_new_tokens: int = 256

    # Allow HuggingFace datasets to execute remote code when loading benchmark datasets
    # (e.g. Winogrande, HellaSwag). Required by most standard benchmark loaders.
    eval_trust_remote_code_datasets: bool = True

    # Re-evaluate iterations even if a cached JSON result already exists.
    # False (default) = skip iterations that already have a .parsed.json result file.
    eval_no_cache: bool = False

    # Automatically run benchmark evaluation after all SPIN training iterations complete.
    # Set to False to skip evaluation and run it separately with evaluate.py.
    eval_run_after_training: bool = True
