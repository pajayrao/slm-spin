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
    # Qwen3-0.6B-Base: 28 layers, hidden 1024, 16 Q / 8 KV heads (head_dim 128),
    # vocab 151,936, LLaMA-style q/k/v/o_proj attention (LoRA targets already match).
    # Requires transformers >= 4.51. Base (pretrain-only) variant chosen deliberately:
    # the SPIN recipe is base → SFT-on-gold → SPIN, and in our results
    # instruction-tuned starting checkpoints never beat their own baseline (0/2 runs)
    # — their existing alignment fights the ultrachat distribution. Switch to
    # "Qwen/Qwen3-0.6B" only to deliberately ablate an instruct/thinking start
    # (its chat template emits <think> blocks; see chat_template_mode below).
    model_name_or_path: str = "Qwen/Qwen3-0.6B-Base"

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
    # Qwen3 caveat: the Qwen3 chat template supports thinking mode (enable_thinking,
    # default True), so synthetic generations may open with a <think>...</think> block
    # while the human ultrachat responses never do. That gives the SPIN discriminator a
    # trivial surface feature to separate chosen from rejected on, weakening the training
    # signal. If synthetic JSONL rows show <think> blocks, either strip them post-
    # generation or pass enable_thinking=False where apply_chat_template is called.
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
    # 1024 (vs the earlier 512) because ultrachat human answers are long: at 512 the
    # chosen side was frequently left-truncated (sometimes into promptless fragments)
    # while the ≤max_new_tokens rejected side never was — a systematic chosen/rejected
    # asymmetry polluting the SPIN margin. If training OOMs at this length, enable
    # gradient_checkpointing rather than shrinking this back.
    # Recommended range for 8 GB GPU: 512–1024.
    max_length: int = 1024

    # Which end of an overlong sequence to truncate.
    # "left"  — drops tokens from the beginning of the prompt (preserves the question tail).
    # "right" — drops tokens from the end of the response (model never sees the full answer).
    truncation_side: str = "left"

    # ── Generation (synthetic response production) ───────────────────────────

    # Number of prompts decoded in a single GPU batch during synthetic generation.
    # KV-cache peak = batch × (max_prompt + max_new_tokens) × layers × KV-heads × head_dim × 2 (K+V) × 2B.
    # Qwen3-0.6B: 28 layers × 8 KV-heads × 128 head_dim × 2 × 2B = 112 KB per token —
    # ~9× more than Qwen2.5-0.5B (24 layers × 2 KV-heads × 64 head_dim = 12 KB/token),
    # so the batch size that was safe for Qwen2.5 is not safe here.
    # batch=32, 640 tokens (256 prompt + 384 new): KV ≈ 2.3 GB + model ~1.2 GB (bf16)
    # = ~3.5 GB — safe on 8 GB. batch=64 would need ~4.6 GB KV — too tight.
    generation_batch_size: int = 32

    # Maximum number of new tokens the model may produce per response.
    # Longer responses create richer training signal but increase generation time linearly.
    # 384 keeps rejected responses length-comparable to the (long) chosen ultrachat
    # answers now that max_length=1024 gives them room; 256 made length itself a
    # trivial chosen/rejected separator.
    # Range: 64–1024. Keep in mind: generation_max_new_tokens ≤ max_length − prompt_length.
    generation_max_new_tokens: int = 384

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
    # Qwen3-0.6B (vocab=151,936, max_length=1024): batch=8 → 8×1024×151936×2B ≈ 2.5 GB
    # worst-case per side — chosen and rejected are scored sequentially, so peak is one
    # side at a time; safe on 8 GB alongside the bf16 model (~1.2 GB). Halve to 4 if
    # scoring OOMs on long-sequence batches.
    ref_logprob_batch_size: int = 4

    # ── SPIN training loop ───────────────────────────────────────────────────

    # Total number of SPIN outer iterations. Each iteration:
    #   1. Uses the current model to generate synthetic responses (the "opponent").
    #   2. Trains a new model to prefer human responses over those synthetic ones.
    # 4 (down from 10): across all 7 completed runs the best checkpoint landed at
    # iteration 0–3 and no run ever recovered past its own peak — iterations beyond
    # ~4 only burned compute. Best-checkpoint selection from the eval table does the
    # rest.
    # Range: 1–10. Typical: 3–5.
    num_iterations: int = 4

    # Number of full passes over the synthetic dataset inside a single SPIN iteration.
    # More epochs = stronger fitting to current synthetic data, but risks overfitting.
    # 2 matches the SPIN paper's per-iteration schedule; watch train/win_rate — if it
    # pins at 1.0 early in epoch 2, drop back to 1.
    # Range: 1–5. Typical: 1–3.
    num_epochs_per_iteration: int = 1

    # Fresh prompts per iteration (Tier-3 experiment; default False = original
    # behaviour where every iteration reuses the same rows). When True, the loaded
    # dataset is partitioned into num_iterations disjoint slices and iteration N
    # trains only on slice N — giving each iteration genuinely new supervision on the
    # chosen side, which is the only lever that adds new information across iterations
    # (the reused-rows default provides none, a leading cause of post-iteration-0
    # decay). To use it meaningfully raise max_data_load so each slice is still large
    # (e.g. max_data_load=200000 with num_iterations=4 → ~50k fresh rows per
    # iteration) and cap the SFT warmup independently via sft_warmup_max_samples so
    # the warmup does not train on all 200k. Slices are deterministic in the loaded
    # (fixed) row order, so resume is unaffected. Start this in a FRESH output_dir.
    resample_prompts_per_iteration: bool = True

    # Hard cap on the total number of records read from the dataset at load time.
    # Applied in load_base_dataset_fixed() before any per-iteration sampling.
    # Use this to bound memory and startup time when the source dataset is very large
    # (e.g. ultrachat_200k has ~200 k rows; setting this to 50 000 loads only the first 50 k).
    # 0 = load the full dataset split.
    # Range: 0 (unlimited) or any positive integer ≤ dataset size.
    max_data_load: int = 200000      

    # Number of dataset rows processed as one atomic checkpoint unit during
    # synthetic generation and ref-logprob scoring. Each batch is saved to
    # iter_{i}_batch_{k:06d}.jsonl before the next batch starts, so a crash
    # loses at most one batch worth of GPU work.
    # Smaller → more frequent saves, lower restart cost.
    # Larger  → fewer file writes, but more work lost per crash.
    # Range: 50–10000. Start at 200 and tune for your restart tolerance.
    data_batch_size: int = 50000

    # λ (lambda) applied in all iterations except the last.
    # Scales the SPIN margin: margin = λ × [(π_θ(chosen) − π_ref(chosen)) − (π_θ(rejected) − π_ref(rejected))].
    # Larger λ = stronger gradient signal, but too large can destabilise training.
    # NOTE: log-probs are per-token averages, so raw margins live in roughly ±(0.5–5)
    # nats — and with rejected_adv_clip=5.0 the margin ceiling is λ × (chosen_adv + 5).
    # λ must keep that ceiling inside the logistic loss's active region: at λ=1 the
    # ceiling margin is ~5 (sigmoid(−5)≈0.007, gradient small but alive); at λ=10 it
    # is ~50 (gradient identically zero — the old λ=10 value only ever "worked"
    # because the pre-fix trainer clipped every update to unit norm, making λ
    # irrelevant). Keep λ ≈ 1–2 with the clip enabled.
    # Range: 0.5–5 with per-token normalization + clipping. Typical: 1.
    lambda_initial: float = 1.0

    # λ used exclusively in the final SPIN iteration (if final_iteration_lambda_only=True).
    # A larger value here applies a stronger final alignment push.
    # NOTE: previously defaulted to 20.0 (40x lambda_initial) — that jump was found to
    # massively amplify likelihood-displacement collapse (see rejected_adv_clip below)
    # on exactly the iteration meant to produce the most stable final checkpoint.
    # Keep this within ~2-4x lambda_initial unless you have evidence the model needs
    # a much stronger final push.
    # Range: 1–10 with per-token normalization.
    lambda_final_iteration: Optional[float] = 2.0

    # When True, lambda_final_iteration replaces lambda_initial only for the very last
    # iteration; all earlier iterations still use lambda_initial.
    # When False, lambda_final_iteration is ignored and lambda_initial is used throughout.
    final_iteration_lambda_only: bool = True

    # Loss function applied to the SPIN margin. All variants are minimised when margin > 0.
    # "logistic"    — softplus(−margin): smooth, never saturates, recommended.
    # "hinge"       — relu(1 − margin): zero loss once margin > 1; hard boundary.
    # "correlation" — (1 − margin): linear penalty; constant gradient, easiest to tune.
    # "exponential" — exp(−margin): very aggressive for negative margins; can cause instability.
    # logistic (was hinge): hinge's dead zone was observed directly in training logs —
    # loss=0 / grad=0 on most steps once margins blew past 1, leaving a handful of
    # unsaturated examples to steer every update while the rejected-side collapse ran
    # silent. The one clearly successful run (Qwen2.5-1.5B, +1.90) also used logistic.
    loss_type: str = "logistic"

    # ── SPIN loss regularisation (anti likelihood-displacement) ────────────────
    # The base SPIN/DPO-style margin loss only depends on
    # (π_θ(chosen)−π_ref(chosen)) − (π_θ(rejected)−π_ref(rejected)). Because that
    # difference can be driven up by either raising chosen or lowering rejected, and
    # lowering rejected is usually the cheaper gradient path, unregularised training
    # tends to crater π_θ(rejected) far more than it raises π_θ(chosen) — a failure
    # mode known as "likelihood displacement". The three knobs below counteract it.

    # Weight of an auxiliary NLL anchor term added directly on the chosen (human)
    # response: loss += sft_alpha * (-mean(pi_chosen_logp)). This gives the optimiser
    # an explicit reward for raising chosen likelihood on its own, independent of the
    # margin, instead of letting it satisfy the margin purely by cratering rejected.
    # 0.0 disables the term (original behaviour).
    # 0.5 (was 0.25): the corrected Qwen3-0.6B run showed the margin was still won
    # ~20x more by rejected-suppression (rejected_adv ~-2.0) than chosen-raising
    # (chosen_adv ~+0.1), and iteration 1 declined on benchmarks despite healthy
    # training loss. A stronger anchor forces more of the improvement onto the
    # chosen side. Raise further (up to ~1.0) if chosen_adv stays near zero.
    # Range: 0.0–1.0. Typical: 0.1–0.5.
    sft_alpha: float = 0.5

    # Maximum magnitude (in nats, per-token-average log-prob units) that the rejected
    # advantage (π_θ(rejected) − π_ref(rejected)) is allowed to contribute to the
    # margin. Values below -rejected_adv_clip are clamped before the margin is formed,
    # so once the model has already moved this far away from the reference on the
    # rejected side, further collapse earns no additional loss reduction.
    # None disables clamping (original behaviour).
    # 2.0 (was 5.0): tightening the clip caps how much margin the optimiser can earn
    # by pushing rejected down, so it must rely on the chosen side once the rejected
    # advantage passes -2.0 nats/token (which the corrected run reached within one
    # iteration). Pairs with the higher sft_alpha above.
    # Range: 2.0–10.0. Typical: 2.0–5.0.
    rejected_adv_clip: Optional[float] = 2.0

    # Weight of a penalty on negative kl_from_ref, i.e. loss += kl_penalty_alpha *
    # relu(-kl_from_ref)**2. kl_from_ref = mean(chosen_adv + rejected_adv) / 2; a
    # sustained negative value means the model is regressing overall relative to the
    # reference rather than improving on both sides, which the rest of this file's
    # docs already call "an alignment alarm". This only fires when kl_from_ref < 0,
    # so it does not penalise the normal, expected direction of drift.
    # 0.0 disables the term (original behaviour).
    # Range: 0.0–1.0. Typical: 0.1.
    kl_penalty_alpha: float = 0.1

    # Fixed-opponent SPIN (Tier-3 experiment; default False = standard SPIN where
    # iteration N's opponent is the iteration-(N-1) model). When True, the OPPONENT —
    # the model that generates the synthetic "rejected" responses and provides the
    # reference log-probs — is frozen at the iteration-0 seed (the SFT-warmed model,
    # or the base model if no warmup) for every iteration, while the trainable policy
    # still continues from the previous iteration's checkpoint. This keeps the
    # negative distribution far from human so the chosen/rejected gap stays large and
    # informative, and avoids the "disprefer your own best model" trap where standard
    # SPIN trains iteration 1 to move away from the (better) iteration-0 model. It is
    # an ALTERNATIVE to resample_prompts_per_iteration, not a complement — enabling
    # both at once conflates two interventions, so run them as separate ablations.
    # Start this in a FRESH output_dir.
    fixed_opponent: bool = False

    # ── Training hyperparameters ─────────────────────────────────────────────

    # Root directory where all run outputs are written (config snapshot, logs, checkpoints).
    output_dir: str = "./spin_outputs"

    # Number of training examples processed per GPU per optimizer step.
    # This is the primary knob for GPU memory: activation memory scales linearly with it.
    # For an 8 GB GPU with a ~1B parameter model: use 1.
    # For a 24 GB GPU: try 4–8.
    # Range: 1–32 (GPU-memory dependent).
    per_device_train_batch_size: int = 1

    # Gradients are accumulated over this many forward passes before one optimizer step.
    # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps.
    # Increase this to compensate when you lower per_device_train_batch_size to fit in memory.
    # 64 × per_device=1 → effective batch 64, matching the SPIN paper, via the
    # memory-cheapest route: raising per_device to 2 instead would double activation
    # memory, which max_length=1024 already doubled (the two-graph SPIN backward holds
    # chosen AND rejected activations simultaneously). Same total FLOPs either way.
    # Range: 1–512. Typical: 32–128.
    gradient_accumulation_steps: int = 16

    # Peak learning rate used during early SPIN iterations (iterations < late_lr_start_iteration).
    # This LR applies to LoRA adapter params only (fp32, effective scale ×α/r=2), not
    # full weights — LoRA preference-tuning convention is 5e-6–5e-5. The previous 1e-6
    # was calibrated against the pre-fix trainer (32×-inflated gradients clipped to
    # unit norm every step); under correct gradient scaling it barely moves rank-16
    # adapters within an iteration. Escalate to 2e-5 if train/margin_mean stays flat
    # through iteration 0; back off if train/kl_from_ref trends strongly negative.
    # Range: 1e-6–5e-5 (LoRA). Typical: 1e-5.
    learning_rate: float = 1e-5

    # Learning rate used from late_lr_start_iteration onward.
    # Smaller than learning_rate to allow fine-grained alignment in later iterations.
    # Range: 1e-6–1e-5 (LoRA). Typical: half the peak LR.
    learning_rate_late: float = 5e-6

    # SPIN iteration index (0-based) at which the LR switches from learning_rate to learning_rate_late.
    # E.g. 1 means iteration 0 uses learning_rate and iterations 1+ use learning_rate_late.
    # 1 (was 2): the self-play signal weakens sharply after iteration 0 (the opponent
    # approaches human quality, so the chosen/rejected gap shrinks). A smaller step
    # from iteration 1 onward reduces the drift that caused iteration 1 to fall below
    # iteration 0's benchmark peak in the corrected run.
    late_lr_start_iteration: int = 1

    # L2 regularisation coefficient applied to weight matrices (not biases or layer norms).
    # 0.0 is standard for supervised fine-tuning. Small values (1e-4) can help generalisation.
    # Range: 0.0–0.1.
    weight_decay: float = 0.01

    # Number of linear LR warmup steps at the beginning of each iteration.
    # Must be small relative to total optimizer steps per SPIN batch.
    # With per_device=1, GA=64, data_batch=50000, epochs=2:
    #   total optimizer steps per batch ≈ (50000/1/64)*2 ≈ 1562
    # So warmup_steps=50 → ~3% warmup — a real ramp matters at the 10× higher peak LR
    # so the first optimizer steps don't shock the zero-initialised lora_B adapters
    # (the old value of 5 was 0.3%, effectively no warmup).
    warmup_steps: int = 50

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
    compile_mode: str = "max-autotune-no-cudagraphs"

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
    # Disabled: the ref model is moved CPU↔GPU around every batch's generation/scoring
    # steps, and compiled artifacts are shape/device-specialised — each round-trip pays
    # recompilation (with max-autotune, minutes of kernel search) for little
    # steady-state benefit on a model that only runs forward passes.
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
    # Qwen3 / Qwen2.5 / SmolLM2 use LLaMA-style attention: "q_proj,k_proj,v_proj,o_proj".
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
    parameter_log_max_tensors: int = 1000

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
    embedding_projector_n_tokens: int = 8192

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
    # Worst case with eval_max_seq_len=2048 and Qwen-scale vocab:
    #   4 × 2048 × 151936 × 2B ≈ 2.5 GB — safe on 8 GB alongside the bf16 model.
    # (8 × 2048 would be ~5 GB — too tight.) Was 1: ~4× slower for no memory benefit.
    eval_batch_size: int = 1

    # Maximum total token length (context + continuation) fed to the model during evaluation.
    # Sequences longer than this are truncated from the left.
    # 2048 (was 1024) because ARC is scored 25-shot: the few-shot prefix alone runs
    # ~1500–2000 tokens, so at 1024 most ARC exemplars (and some 10-shot HellaSwag
    # context) were silently truncated away — systematically depressing those scores
    # in every previous eval. NOTE: changing this invalidates comparison with all
    # cached .parsed.json results — re-evaluate base_model and any checkpoints you
    # intend to report under the new setting (delete the caches or use --no-cache).
    eval_max_seq_len: int = 2048

    # Apply torch.compile() to each model before evaluation. Disabled by default:
    # eval feeds variable-length sequences, so torch.compile recompiles per shape and
    # inflates GPU memory (and rebuilds Triton kernels) for little speedup — it is the
    # main cause of eval OOM on small GPUs. Training compilation is controlled separately
    # by compile_model and is unaffected by this flag.
    eval_compile_model: bool = False

    # Maximum new tokens generated per response in the GSM8k benchmark (generation task).
    eval_gsm8k_max_new_tokens: int = 256

    # Allow HuggingFace datasets to execute remote code when loading benchmark datasets
    # (e.g. Winogrande, HellaSwag). Required by most standard benchmark loaders.
    eval_trust_remote_code_datasets: bool = True

    # Re-evaluate iterations even if a cached JSON result already exists.
    # False (default) = skip iterations that already have a .parsed.json result file.
    eval_no_cache: bool = False

    # Automatically run benchmark evaluation (evaluate.run_eval) as soon as each SPIN
    # iteration's checkpoint is saved, instead of waiting until all iterations finish.
    # Already-evaluated iterations are skipped via the same .parsed.json cache used by
    # the standalone evaluate.py, so this adds no overhead on resume.
    eval_run_after_training: bool = True

    # ── SFT warmup (optional — reproduces the SPIN paper's precondition) ──────────

    # When True, run one supervised fine-tuning pass over the (prompt, response) rows
    # BEFORE the SPIN loop and use that checkpoint as iteration 0's starting model
    # instead of model_name_or_path. This reproduces the paper's setup
    # (zephyr-7b-sft-full = a base model SFT'd on the SPIN dataset), so SPIN starts
    # already fitted to p_data and *sharpens* the model rather than relocating it.
    # Lets any base model be plugged in:  base → SFT-on-gold → SPIN.
    sft_warmup_enabled: bool = True

    # Directory where the warmed-up model + tokenizer are saved. Reused as the
    # iteration-0 base on resume (skipped if a completed checkpoint already exists).
    sft_warmup_dir: str = "./spin_outputs/sft_warmup"

    # Epochs for the warmup SFT pass. The paper uses 1 — enough to fit the model to
    # p_data while leaving the residual "quality gap" that SPIN then exploits.
    sft_warmup_epochs: float = 1.0

    # Peak learning rate for the warmup SFT pass (independent of the SPIN LRs).
    sft_warmup_learning_rate: float = 2e-5

    # Cap on the number of (prompt, response) rows used for warmup SFT.
    # 0 = use all loaded rows (already bounded by max_data_load).
    sft_warmup_max_samples: int = 0

    # Cache the tokenized warmup dataset to a .pt file so re-runs skip tokenization.
    sft_warmup_cache_tokenized: bool = True

