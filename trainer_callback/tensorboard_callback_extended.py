import math
import os
import time
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from spin_config import SPINConfig
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# TensorBoardCallbackExtended
# ─────────────────────────────────────────────────────────────────────────────

class TensorBoardCallbackExtended(TrainerCallback):
    """
    Per-iteration TensorBoard logger. Each SPIN iteration writes to its own
    subdirectory (iter_dir/tb_logs) so you can compare iterations side-by-side
    in TensorBoard by selecting multiple runs.

    Metrics written:

    train/* (every logged step):
      train/loss              — SPIN margin loss value
      train/margin_mean       — mean SPIN margin across the batch
      train/margin_std        — std-dev of SPIN margin (spread of alignment signal)
      train/win_rate          — fraction of batch examples where margin > 0
      train/pi_chosen_logp    — mean log-prob of human response under π_θ
      train/pi_rejected_logp  — mean log-prob of synthetic response under π_θ
      train/ref_chosen_logp   — mean log-prob of human response under π_ref (frozen)
      train/ref_rejected_logp — mean log-prob of synthetic response under π_ref (frozen)
      train/logp_gap          — pi_chosen_logp − pi_rejected_logp (should grow > 0)
      train/kl_from_ref       — mean KL divergence from reference model
      train/spin_lambda       — λ used this iteration
      train/learning_rate     — current LR from the scheduler
      train/grad_global_norm  — L2 norm of all gradients (explosion detector)
      train/weight_global_norm— L2 norm of all trainable parameters
      train/weight_drift      — L2 norm of total weight change since iteration start
      train/throughput_sps    — training samples per second (throughput)
      train/perplexity        — exp(loss) — more interpretable than raw loss
      train/alignment_accuracy— 1.0 when margin_mean > 0

    system/* (every logged step):
      system/gpu_alloc_mb     — active GPU allocation
      system/gpu_reserved_mb  — reserved GPU allocation
      system/gpu_util_pct     — GPU compute utilisation percentage
      system/cpu_rss_mb       — CPU RSS

    weights/* (every parameter_log_interval steps, per layer):
      weights/mean, std, norm, absmax, delta_norm

    gradients/* (every parameter_log_interval steps, per layer):
      gradients/norm, absmax, histogram

    epoch/* (at end of each epoch):
      epoch/loss_mean, loss_min, loss_max
      epoch/margin_mean, margin_final
      epoch/win_rate_mean
      epoch/logp_gap_mean

    iteration_summary/* (at on_train_end):
      iteration_summary/final_loss
      iteration_summary/final_margin
      iteration_summary/final_logp_gap
      iteration_summary/total_weight_drift
    """

    def __init__(self, log_dir: str, cfg: SPINConfig, log_histograms: bool = False,
                 tokenizer=None, iteration: int = 0):
        self.writer = SummaryWriter(log_dir)
        self.cfg = cfg
        self.log_histograms = log_histograms
        self._tokenizer = tokenizer
        self._spin_iteration = iteration

        # Populated in on_train_begin; used to compute weight drift during training
        self.initial_params: dict = {}

        # Accumulators reset each epoch
        self._epoch_losses:    list = []
        self._epoch_margins:   list = []
        self._epoch_gaps:      list = []
        self._epoch_win_rates: list = []

        # Captured from on_log so on_train_end can include it in hparams
        self._spin_lambda_val: float = 0.0

        # Throughput tracking
        self._step_start_time: float = 0.0

        # pynvml handle for GPU utilization % (None if pynvml unavailable)
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Snapshot weights at iteration start for drift tracking
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.detach().float().cpu().clone()

        # Log a text summary of the training config for reference
        cfg_text = "\n".join(f"    {k}: {v}" for k, v in vars(self.cfg).items())
        self.writer.add_text("config/spin_config", f"```\n{cfg_text}\n```", 0)

        # GRAPHS tab: trace the model with a tiny dummy input.
        self._log_model_graph(model)

        # PROJECTOR tab: log token embeddings at iteration start
        self._log_token_embeddings(model, step=0, tag="embeddings/tokens_iter_start")

        # Custom scalars layout: groups related metrics onto shared charts.
        self.writer.add_custom_scalars({
            "Alignment": {
                "LogProbs (π_θ)":        ["Multiline", ["train/pi_chosen_logp", "train/pi_rejected_logp"]],
                "LogProbs (π_ref)":       ["Multiline", ["train/ref_chosen_logp", "train/ref_rejected_logp"]],
                "LogP Gap vs Margin":     ["Multiline", ["train/logp_gap", "train/margin_mean"]],
                "Win Rate":               ["Multiline", ["train/win_rate"]],
                "KL from Reference":      ["Multiline", ["train/kl_from_ref"]],
            },
            "Loss": {
                "Train Loss":  ["Multiline", ["train/loss"]],
                "Perplexity":  ["Multiline", ["train/perplexity"]],
                "Margin Std":  ["Multiline", ["train/margin_std"]],
            },
            "System": {
                "GPU Memory (MB)": ["Multiline", ["system/gpu_alloc_mb", "system/gpu_reserved_mb"]],
                "GPU Util %":      ["Multiline", ["system/gpu_util_pct"]],
                "CPU RSS (MB)":    ["Multiline", ["system/cpu_rss_mb"]],
            },
            "Gradients": {
                "Global Norms":    ["Multiline", ["train/grad_global_norm", "train/weight_global_norm"]],
                "Weight Drift":    ["Multiline", ["train/weight_drift"]],
            },
        })
        self.writer.flush()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Intercepts every self.log() call from SPINTrainer.training_step.
        Writes each logged value under the train/ namespace and computes
        derived metrics (logp_gap, throughput).
        """
        if logs is None or self.writer is None:
            return
        step = state.global_step

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, step)

        # Capture spin_lambda for use in on_train_end hparams
        if "spin_lambda" in logs:
            self._spin_lambda_val = float(logs["spin_lambda"])

        # logp_gap: positive means π_θ prefers human over synthetic — the goal of SPIN
        chosen   = logs.get("pi_chosen_logp")
        rejected = logs.get("pi_rejected_logp")
        if chosen is not None and rejected is not None:
            gap = chosen - rejected
            self.writer.add_scalar("train/logp_gap", gap, step)
            self._epoch_gaps.append(gap)

        loss   = logs.get("loss") or logs.get("train_loss")
        margin = logs.get("margin_mean")
        wr     = logs.get("win_rate")
        if loss   is not None: self._epoch_losses.append(loss)
        if margin is not None: self._epoch_margins.append(margin)
        if wr     is not None: self._epoch_win_rates.append(wr)

        # Perplexity: exp(loss) — more interpretable than raw cross-entropy for LLMs
        if loss is not None:
            self.writer.add_scalar("train/perplexity", math.exp(min(loss, 20)), step)

        # Alignment accuracy: 1.0 when margin_mean > 0 (model already prefers human responses)
        if margin is not None:
            self.writer.add_scalar("train/alignment_accuracy", float(margin > 0), step)

        # Throughput: samples per second (one step = per_device_train_batch_size samples)
        if self._step_start_time > 0:
            elapsed = time.time() - self._step_start_time
            if elapsed > 0:
                sps = args.per_device_train_batch_size / elapsed
                self.writer.add_scalar("train/throughput_sps", sps, step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.writer is None or model is None:
            return
        step = state.global_step

        # ── System memory ────────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.writer.add_scalar("system/gpu_alloc_mb",    torch.cuda.memory_allocated() / 1024 ** 2, step)
            self.writer.add_scalar("system/gpu_reserved_mb", torch.cuda.memory_reserved()  / 1024 ** 2, step)
        try:
            import psutil
            self.writer.add_scalar("system/cpu_rss_mb", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, step)
        except ImportError:
            pass

        # ── GPU utilization % ────────────────────────────────────────────────
        # Try pynvml first (more accurate); fall back to torch.cuda.utilization()
        gpu_util = None
        if self._nvml_handle is not None:
            try:
                import pynvml
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
            except Exception:
                pass
        if gpu_util is None and torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
            except Exception:
                pass
        if gpu_util is not None:
            self.writer.add_scalar("system/gpu_util_pct", gpu_util, step)

        # ── Global gradient norm ─────────────────────────────────────────────
        grad_sq_sum = sum(
            p.grad.detach().float().norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        self.writer.add_scalar("train/grad_global_norm", grad_sq_sum ** 0.5, step)

        # ── Per-parameter stats (throttled to parameter_log_interval) ────────
        # HuggingFace Trainer increments global_step before calling on_step_end,
        # so step starts at 1; check step == 1 to fire on the very first optimizer step.
        if not self._should_log_params(step):
            self.writer.flush()
            return

        total_weight_sq = 0.0
        total_delta_sq  = 0.0
        logged = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if logged >= self.cfg.parameter_log_max_tensors:
                break

            w   = param.detach().float().cpu()
            tag = name.replace(".", "/")
            total_weight_sq += w.norm().item() ** 2

            if self.log_histograms or self.cfg.log_parameter_histograms:
                self.writer.add_histogram(f"weights/{tag}", w, step)

            if self.cfg.log_parameter_scalars:
                self.writer.add_scalar(f"weights/mean/{tag}",   w.mean().item(),      step)
                self.writer.add_scalar(f"weights/std/{tag}",    w.std().item(),       step)
                self.writer.add_scalar(f"weights/norm/{tag}",   w.norm().item(),      step)
                self.writer.add_scalar(f"weights/absmax/{tag}", w.abs().max().item(), step)

            # Weight drift from the start of this SPIN iteration
            if name in self.initial_params:
                delta = w - self.initial_params[name]
                total_delta_sq += delta.norm().item() ** 2
                if self.cfg.log_parameter_scalars:
                    self.writer.add_scalar(f"weight_delta/norm/{tag}", delta.norm().item(), step)
                    self.writer.add_scalar(f"weight_delta/mean/{tag}", delta.mean().item(), step)

            # Gradient stats for this parameter
            if param.grad is not None:
                g = param.grad.detach().float().cpu()
                if self.cfg.log_gradient_histograms:
                    self.writer.add_histogram(f"gradients/{tag}", g, step)
                if self.cfg.log_parameter_scalars:
                    self.writer.add_scalar(f"gradients/norm/{tag}",   g.norm().item(),      step)
                    self.writer.add_scalar(f"gradients/absmax/{tag}", g.abs().max().item(), step)
                # Grad-to-weight ratio: detects vanishing/exploding gradients relative to parameter scale
                self.writer.add_scalar(
                    f"layer_health/grad_weight_ratio/{tag}",
                    g.norm().item() / (w.norm().item() + 1e-8),
                    step,
                )

            logged += 1

        # Global aggregates: useful for a single high-level view without per-layer noise
        self.writer.add_scalar("train/weight_global_norm", total_weight_sq ** 0.5, step)
        self.writer.add_scalar("train/weight_drift",        total_delta_sq  ** 0.5, step)
        self.writer.flush()

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Logs aggregate statistics over all steps in the completed epoch.
        The x-axis is epoch number, making it easy to see loss trends epoch-over-epoch
        within a single SPIN iteration.
        """
        epoch = int(state.epoch) if state.epoch is not None else 0

        if self._epoch_losses:
            self.writer.add_scalar("epoch/loss_mean",  sum(self._epoch_losses) / len(self._epoch_losses), epoch)
            self.writer.add_scalar("epoch/loss_min",   min(self._epoch_losses), epoch)
            self.writer.add_scalar("epoch/loss_max",   max(self._epoch_losses), epoch)
            self.writer.add_scalar("epoch/loss_final", self._epoch_losses[-1],  epoch)
            self.writer.add_histogram("epoch/loss_distribution", torch.tensor(self._epoch_losses), epoch)

        if self._epoch_margins:
            self.writer.add_scalar("epoch/margin_mean",  sum(self._epoch_margins) / len(self._epoch_margins), epoch)
            self.writer.add_scalar("epoch/margin_final", self._epoch_margins[-1], epoch)

        if self._epoch_win_rates:
            self.writer.add_scalar("epoch/win_rate_mean",  sum(self._epoch_win_rates) / len(self._epoch_win_rates), epoch)
            self.writer.add_scalar("epoch/win_rate_final", self._epoch_win_rates[-1], epoch)

        if self._epoch_gaps:
            self.writer.add_scalar("epoch/logp_gap_mean",  sum(self._epoch_gaps) / len(self._epoch_gaps), epoch)
            self.writer.add_scalar("epoch/logp_gap_final", self._epoch_gaps[-1], epoch)

        # PR CURVES tab: Precision-Recall curve for alignment accuracy.
        # Each data point is one logged batch. Label = 1 when the batch had positive
        # alignment (margin_mean > 0); score = sigmoid(margin_mean) as confidence.
        # The curve shows how reliably the model's margin score predicts alignment.
        if self.cfg.log_pr_curves and len(self._epoch_margins) > 1:
            labels = torch.tensor([1.0 if m > 0 else 0.0 for m in self._epoch_margins])
            scores = torch.sigmoid(torch.tensor(self._epoch_margins, dtype=torch.float32))
            self.writer.add_pr_curve("train/alignment_pr_curve", labels, scores, epoch)

        # Reset accumulators for the next epoch
        self._epoch_losses.clear()
        self._epoch_margins.clear()
        self._epoch_gaps.clear()
        self._epoch_win_rates.clear()
        self.writer.flush()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """
        Writes a final summary at the end of training for this SPIN iteration.
        """
        if model is not None and self.initial_params:
            total_delta_sq = sum(
                (param.detach().float().cpu() - self.initial_params[name]).norm().item() ** 2
                for name, param in model.named_parameters()
                if param.requires_grad and name in self.initial_params
            )
            self.writer.add_scalar("iteration_summary/total_weight_drift", total_delta_sq ** 0.5, state.global_step)

        # Scan log_history in reverse for the last step that actually contains training metrics.
        # The final entry is often a timing summary (train_runtime, etc.) without loss values.
        last_metrics: dict = {}
        for entry in reversed(state.log_history):
            if "loss" in entry or "train_loss" in entry:
                last_metrics = entry
                break

        for key in ("loss", "train_loss", "margin_mean", "pi_chosen_logp", "pi_rejected_logp",
                    "win_rate", "kl_from_ref"):
            val = last_metrics.get(key)
            if val is not None:
                canonical = key if key != "train_loss" else "loss"
                self.writer.add_scalar(f"iteration_summary/final_{canonical}", val, state.global_step)

        if "pi_chosen_logp" in last_metrics and "pi_rejected_logp" in last_metrics:
            self.writer.add_scalar(
                "iteration_summary/final_logp_gap",
                last_metrics["pi_chosen_logp"] - last_metrics["pi_rejected_logp"],
                state.global_step,
            )

        # PROJECTOR tab: snapshot token embeddings at iteration end.
        if model is not None:
            self._log_token_embeddings(model, step=state.global_step, tag="embeddings/tokens_iter_end")

        # HParams plugin: links hyperparameters to final metrics for cross-run comparison.
        # spin_iteration and spin_lambda vary across runs, making the Parallel Coordinates
        # and Scatter Plot views informative.
        hparam_dict = {
            "learning_rate":    args.learning_rate,
            "batch_size":       args.per_device_train_batch_size,
            "num_epochs":       args.num_train_epochs,
            "spin_iteration":   self._spin_iteration,
            "spin_lambda":      self._spin_lambda_val,
        }
        hparam_metrics: dict = {}
        loss_val = last_metrics.get("loss") or last_metrics.get("train_loss")
        if loss_val is not None:
            hparam_metrics["hparam/final_loss"] = loss_val
        margin_val = last_metrics.get("margin_mean")
        if margin_val is not None:
            hparam_metrics["hparam/final_margin"] = margin_val
        wr_val = last_metrics.get("win_rate")
        if wr_val is not None:
            hparam_metrics["hparam/final_win_rate"] = wr_val
        if "pi_chosen_logp" in last_metrics and "pi_rejected_logp" in last_metrics:
            hparam_metrics["hparam/final_logp_gap"] = (
                last_metrics["pi_chosen_logp"] - last_metrics["pi_rejected_logp"]
            )
        kl_val = last_metrics.get("kl_from_ref")
        if kl_val is not None:
            hparam_metrics["hparam/final_kl_from_ref"] = kl_val

        if hparam_metrics:
            try:
                self.writer.add_hparams(hparam_dict, hparam_metrics)
            except Exception as e:
                logger.warning(f"add_hparams failed: {e}")

        self.writer.flush()
        self.writer.close()

    def _log_model_graph(self, model):
        """Write the architecture graph to the per-iteration GRAPHS tab.

        Uses a traceable graph module built from model.config to avoid trace
        failures from flash-attn / SDPA backends and HuggingFace control flow.
        The authoritative copy (written to tb_global) comes from
        SPINIterationSummaryCallback._write_graph_to_global on iteration 0.
        """
        if not self.cfg.log_model_graph or model is None:
            return
        try:
            from trainer_callback.spin_iteration_summary_callback import SPINIterationSummaryCallback
            graph_model = SPINIterationSummaryCallback._build_causal_lm_graph(model)
            graph_model.eval()
            dummy_ids  = torch.zeros(1, 8, dtype=torch.long)
            dummy_mask = torch.ones(1, 8, dtype=torch.long)
            with torch.no_grad():
                self.writer.add_graph(graph_model, (dummy_ids, dummy_mask), use_strict_trace=True)
            self.writer.flush()
            logger.info("[GRAPHS] Architecture graph written to per-iteration TensorBoard run.")
        except Exception as e:
            logger.error(f"[GRAPHS] Graph trace failed in per-iteration writer: "
                         f"{type(e).__name__}: {e}")

    def _log_token_embeddings(self, model, step: int, tag: str):
        """Extract and log the token embedding matrix (PROJECTOR tab)."""
        if not self.cfg.log_embedding_projector or model is None:
            return
        try:
            # Walk common embedding layer names across LLaMA / Mistral / GPT-2 / Falcon
            inner = getattr(model, "model", model)
            embed_layer = (
                getattr(inner, "embed_tokens", None)
                or getattr(inner, "wte", None)
                or getattr(getattr(inner, "embeddings", object()), "word_embeddings", None)
            )
            if embed_layer is None:
                logger.warning("Could not find embedding layer; PROJECTOR skipped.")
                return

            embed = embed_layer.weight.detach().float().cpu()
            n = min(self.cfg.embedding_projector_n_tokens, embed.size(0))
            embed_subset = embed[:n]

            # Metadata: use decoded token strings when a tokenizer is available,
            # otherwise fall back to plain token-ID strings.
            if self._tokenizer is not None:
                metadata = []
                for i in range(n):
                    tok = self._tokenizer.decode([i])
                    metadata.append(tok.strip() or f"<tok_{i}>")
            else:
                metadata = [str(i) for i in range(n)]

            self.writer.add_embedding(embed_subset, metadata=metadata, global_step=step, tag=tag)
            logger.info(f"Token embeddings ({n} tokens) logged to TensorBoard PROJECTOR tab (tag={tag}).")
        except Exception as e:
            logger.warning(f"Could not log token embeddings: {e}")

    def _should_log_params(self, step: int) -> bool:
        # HuggingFace Trainer increments global_step before calling on_step_end,
        # so the first call has step=1; check step==1 to always log on the first optimizer step.
        return step == 1 or step % self.cfg.parameter_log_interval == 0
