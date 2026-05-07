import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from spin_config import SPINConfig

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SPINIterationSummaryCallback
# ─────────────────────────────────────────────────────────────────────────────

class SPINIterationSummaryCallback(TrainerCallback):
    """
    Cross-iteration TensorBoard logger.

    Created ONCE before the outer SPIN loop and reused across all iterations.
    Uses the SPIN iteration index as the x-axis, so every chart shows how the
    model improves from iteration 0 → 1 → 2 → ...

    Call  set_iteration(i)  before each trainer.train() so the callback knows
    which SPIN iteration it is recording.
    Call  close()  after the loop ends to flush and close the writer.

    Metrics written (spin_progress/* namespace):

    Loss progression:
      spin_progress/final_loss      — loss at the last logged step of the iteration
      spin_progress/mean_loss       — average loss over the full iteration
      spin_progress/min_loss        — best (lowest) loss seen in the iteration

    Alignment signal:
      spin_progress/final_margin    — SPIN margin at the last step (higher = better alignment)
      spin_progress/mean_margin     — average margin over the iteration
      spin_progress/final_logp_gap  — pi_chosen − pi_rejected at the last step (should grow)
      spin_progress/final_pi_chosen_logp  — how likely the model is to produce human responses
      spin_progress/final_pi_rejected_logp— how likely the model is to produce its old responses

    Model drift:
      spin_progress/weight_drift_from_iter_start — how much the model changed within this iteration
      spin_progress/weight_drift_from_base_model  — cumulative drift from the very first checkpoint

    Training stats:
      spin_progress/total_steps     — optimizer steps taken in this iteration
      spin_progress/final_lr        — learning rate at the end of this iteration
    """

    def __init__(self, log_dir: str, cfg: SPINConfig = None):
        """Args:
            log_dir: directory for the global (cross-iteration) TensorBoard writer.
            cfg:     SPIN configuration; used to gate optional features (graph logging, etc.).
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        logger.info(f"SPINIterationSummaryCallback: global TensorBoard writer opened at {log_dir}. "
                    f"Tracks cross-iteration trends (loss, margin, win_rate, weight drift).")
        self.cfg = cfg
        self.spin_iteration: int = 0

        # Captured once in iteration 0; used to measure cumulative drift thereafter
        self.base_model_params: dict = {}

        # Per-iteration accumulators reset by set_iteration()
        self._losses:       list = []
        self._margins:      list = []
        self._win_rates:    list = []
        self._chosen_lps:   list = []
        self._rejected_lps: list = []
        self._kl_vals:      list = []
        self._final_lr:     float = 0.0
        self._spin_lambda:  float = 0.0
        self._dataset_size: int = 0

        # Snapshot of weights at the start of this iteration
        self._iter_start_params: dict = {}

    def set_iteration(self, iteration: int, spin_lambda: float = 0.0, dataset_size: int = 0):
        """Call this before each trainer.train() to set the SPIN iteration index."""
        self.spin_iteration = iteration
        self._spin_lambda = spin_lambda
        self._dataset_size = dataset_size
        self._losses.clear()
        self._margins.clear()
        self._win_rates.clear()
        self._chosen_lps.clear()
        self._rejected_lps.clear()
        self._kl_vals.clear()
        self._final_lr = 0.0
        self._iter_start_params.clear()
        logger.info(f"SPINIterationSummaryCallback.set_iteration(): iteration={iteration}, "
                    f"spin_lambda={spin_lambda}, dataset_size={dataset_size}. "
                    f"Accumulators cleared — ready to record metrics for this iteration.")

    # ── Architecture graph ─────────────────────────────────────────────────────

    @staticmethod
    def _build_causal_lm_graph(model):
        """
        Build a traceable CausalLM graph module from model.config.

        Uses only standard PyTorch ops so torch.jit.trace succeeds regardless
        of the real model's attention backend (flash-attn, SDPA, etc.).
        Represents the architectural shape: embedding → N × decoder layers → LM head.
        """
        c = model.config
        vocab = getattr(c, 'vocab_size',          32000)
        d = getattr(c, 'hidden_size',           768)
        n_lay = getattr(c, 'num_hidden_layers',      12)
        n_h = getattr(c, 'num_attention_heads',    12)
        ffn = getattr(c, 'intermediate_size', d * 4)

        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                head_dim = max(1, d // n_h)
                self.q = torch.nn.Linear(d, d,   bias=False)
                self.k = torch.nn.Linear(d, d,   bias=False)
                self.v = torch.nn.Linear(d, d,   bias=False)
                self.o = torch.nn.Linear(d, d,   bias=False)
                self.gate = torch.nn.Linear(d, ffn, bias=False)
                self.up = torch.nn.Linear(d, ffn, bias=False)
                self.down = torch.nn.Linear(ffn, d, bias=False)
                self.n1 = torch.nn.LayerNorm(d)
                self.n2 = torch.nn.LayerNorm(d)
                self._hd = head_dim
                self._nh = n_h

            def forward(self, x):
                B, L, D = x.shape
                h = self.n1(x)
                q = self.q(h).view(B, L, self._nh, self._hd).transpose(1, 2)
                k = self.k(h).view(B, L, self._nh, self._hd).transpose(1, 2)
                v = self.v(h).view(B, L, self._nh, self._hd).transpose(1, 2)
                a = (q @ k.transpose(-2, -1)) * (self._hd ** -0.5)
                a = a.softmax(dim=-1)
                out = (a @ v).transpose(1, 2).contiguous().view(B, L, D)
                x = x + self.o(out)
                h2 = self.n2(x)
                x = x + self.down(
                    torch.nn.functional.silu(self.gate(h2)) * self.up(h2)
                )
                return x

        class _CausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab, d)
                self.layers = torch.nn.ModuleList(
                    [_Layer() for _ in range(n_lay)])
                self.norm = torch.nn.LayerNorm(d)
                self.lm_head = torch.nn.Linear(d, vocab, bias=False)

            def forward(self, input_ids, attention_mask):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(self.norm(x))

        return _CausalLM()

    def _write_graph_to_global(self, model):
        """Trace the architecture graph and write it to the global tb_global writer."""
        if not self.cfg.log_model_graph:
            return
        try:
            graph_model = self._build_causal_lm_graph(model)
            graph_model.eval()
            dummy_ids = torch.zeros(1, 8, dtype=torch.long)
            dummy_mask = torch.ones(1, 8, dtype=torch.long)
            with torch.no_grad():
                self.writer.add_graph(
                    graph_model, (dummy_ids, dummy_mask), use_strict_trace=True)
            self.writer.flush()
            logger.info(
                "[GRAPHS] Architecture graph written to tb_global TensorBoard run.")
        except Exception as e:
            logger.error(
                f"[GRAPHS] Graph trace failed: {type(e).__name__}: {e}")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Snapshot iteration-start weights for drift tracking; write the architecture graph on iter 0."""
        if model is None:
            logger.warning(
                "SPINIterationSummaryCallback.on_train_begin: model is None — skipping weight snapshot.")
            return

        trainable = [(n, p)
                     for n, p in model.named_parameters() if p.requires_grad]
        for name, param in trainable:
            self._iter_start_params[name] = param.detach(
            ).float().cpu().clone()
        logger.info(f"SPINIterationSummaryCallback.on_train_begin: "
                    f"snapshotted {len(self._iter_start_params)} trainable param tensors "
                    f"at iteration {self.spin_iteration} start (for drift tracking).")

        if self.spin_iteration == 0 and not self.base_model_params:
            self.base_model_params = {k: v.clone()
                                      for k, v in self._iter_start_params.items()}
            logger.info("  Base model weight snapshot captured at iteration 0 "
                        "(used for cumulative drift from the original checkpoint).")
            self._write_graph_to_global(model)

        # Log a text card showing which SPIN iteration is starting
        self.writer.add_text(
            "spin_iterations/log",
            f"**Iteration {self.spin_iteration} started** — "
            f"LR={args.learning_rate:.2e}, "
            f"epochs={args.num_train_epochs}, "
            f"batch={args.per_device_train_batch_size}",
            self.spin_iteration,
        )
        self.writer.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Accumulate step-level metrics; summarised at on_train_end."""
        if logs is None:
            return
        loss_val = logs.get("loss") or logs.get("train_loss")
        if loss_val is not None:
            self._losses.append(loss_val)
        if "margin_mean" in logs:
            self._margins.append(logs["margin_mean"])
        if "win_rate" in logs:
            self._win_rates.append(logs["win_rate"])
        if "pi_chosen_logp" in logs:
            self._chosen_lps.append(logs["pi_chosen_logp"])
        if "pi_rejected_logp" in logs:
            self._rejected_lps.append(logs["pi_rejected_logp"])
        if "kl_from_ref" in logs:
            self._kl_vals.append(logs["kl_from_ref"])
        if "learning_rate" in logs:
            self._final_lr = logs["learning_rate"]
        if "spin_lambda" in logs:
            self._spin_lambda = logs["spin_lambda"]

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """
        Write the cross-iteration summary. Called at the end of trainer.train()
        for each SPIN iteration. Uses self.spin_iteration as the global step so
        all iterations appear on the same time axis in TensorBoard.
        """
        i = self.spin_iteration
        final_loss = self._losses[-1] if self._losses else float("nan")
        final_margin = self._margins[-1] if self._margins else float("nan")
        final_wr = self._win_rates[-1] if self._win_rates else float("nan")
        logger.info(f"SPINIterationSummaryCallback.on_train_end: writing cross-iteration summary "
                    f"for iteration {i} (global TB step={i}).")
        logger.info(f"  Accumulated steps logged: losses={len(self._losses)}, "
                    f"margins={len(self._margins)}, win_rates={len(self._win_rates)}.")
        logger.info(f"  Final metrics — loss={final_loss:.4f}, margin={final_margin:.4f}, "
                    f"win_rate={final_wr:.3f}.")

        # ── Loss ────────────────────────────────────────────────────────────
        if self._losses:
            self.writer.add_scalar(
                "spin_progress/final_loss", self._losses[-1],                      i)
            self.writer.add_scalar(
                "spin_progress/mean_loss",  sum(self._losses)/len(self._losses),   i)
            self.writer.add_scalar(
                "spin_progress/min_loss",   min(self._losses),                     i)

        # ── Alignment margin ─────────────────────────────────────────────────
        if self._margins:
            self.writer.add_scalar(
                "spin_progress/final_margin", self._margins[-1],                       i)
            self.writer.add_scalar(
                "spin_progress/mean_margin",  sum(self._margins)/len(self._margins),   i)

        # ── Log-probability gap ───────────────────────────────────────────────
        if self._chosen_lps:
            self.writer.add_scalar(
                "spin_progress/final_pi_chosen_logp", self._chosen_lps[-1], i)
        if self._rejected_lps:
            self.writer.add_scalar(
                "spin_progress/final_pi_rejected_logp", self._rejected_lps[-1], i)
        if self._chosen_lps and self._rejected_lps:
            gap = self._chosen_lps[-1] - self._rejected_lps[-1]
            self.writer.add_scalar("spin_progress/final_logp_gap", gap, i)
            # Mean gap over the iteration — more stable than the final value
            mean_gap = (
                sum(c - r for c, r in zip(self._chosen_lps, self._rejected_lps))
                / len(self._chosen_lps)
            )
            self.writer.add_scalar("spin_progress/mean_logp_gap", mean_gap, i)

        # ── Win rate ─────────────────────────────────────────────────────────
        if self._win_rates:
            self.writer.add_scalar(
                "spin_progress/final_win_rate", self._win_rates[-1],                        i)
            self.writer.add_scalar(
                "spin_progress/mean_win_rate",  sum(self._win_rates)/len(self._win_rates),  i)

        # ── KL from reference ────────────────────────────────────────────────
        if self._kl_vals:
            self.writer.add_scalar(
                "spin_progress/final_kl_from_ref", self._kl_vals[-1],                       i)
            self.writer.add_scalar(
                "spin_progress/mean_kl_from_ref",  sum(self._kl_vals)/len(self._kl_vals),   i)

        # ── Training stats ───────────────────────────────────────────────────
        self.writer.add_scalar(
            "spin_progress/total_steps", state.global_step, i)
        if self._final_lr:
            self.writer.add_scalar("spin_progress/final_lr", self._final_lr, i)
        if self._spin_lambda:
            self.writer.add_scalar(
                "spin_progress/spin_lambda", self._spin_lambda, i)
        if self._dataset_size:
            self.writer.add_scalar(
                "spin_progress/dataset_size", self._dataset_size, i)

        # ── Weight drift ─────────────────────────────────────────────────────
        if model is not None:
            iter_drift_sq = 0.0
            base_drift_sq = 0.0

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                w = param.detach().float().cpu()
                if name in self._iter_start_params:
                    iter_drift_sq += (w -
                                      self._iter_start_params[name]).norm().item() ** 2
                if name in self.base_model_params:
                    base_drift_sq += (w -
                                      self.base_model_params[name]).norm().item() ** 2

            # Drift within this iteration: measures how much the model changed in this round
            self.writer.add_scalar(
                "spin_progress/weight_drift_from_iter_start", iter_drift_sq ** 0.5, i)

            # Cumulative drift from the base model: measures total alignment shift across all iterations
            if self.base_model_params:
                self.writer.add_scalar(
                    "spin_progress/weight_drift_from_base_model", base_drift_sq ** 0.5, i)

            # Cosine similarity between iteration-start and iteration-end weights.
            # 1.0 = no change; lower values = larger directional shift this iteration.
            current = {n: p.detach().float().cpu()
                       for n, p in model.named_parameters() if p.requires_grad}
            names = [n for n in self._iter_start_params if n in current]
            if names:
                flat_start = torch.cat(
                    [self._iter_start_params[n].flatten() for n in names])
                flat_end = torch.cat([current[n].flatten() for n in names])
                cos_sim = torch.nn.functional.cosine_similarity(
                    flat_start.unsqueeze(0), flat_end.unsqueeze(0)
                ).item()
                self.writer.add_scalar(
                    "spin_progress/cosine_sim_to_iter_start", cos_sim, i)

        # ── Text summary card ────────────────────────────────────────────────
        summary_lines = [f"## SPIN Iteration {i} Summary"]
        if self._losses:
            summary_lines.append(
                f"- Loss: final={self._losses[-1]:.4f}, mean={sum(self._losses)/len(self._losses):.4f}, min={min(self._losses):.4f}")
        if self._margins:
            summary_lines.append(
                f"- Margin: final={self._margins[-1]:.4f}, mean={sum(self._margins)/len(self._margins):.4f}")
        if self._win_rates:
            summary_lines.append(
                f"- Win rate: final={self._win_rates[-1]:.3f}, mean={sum(self._win_rates)/len(self._win_rates):.3f}")
        if self._chosen_lps and self._rejected_lps:
            summary_lines.append(
                f"- LogP gap: {self._chosen_lps[-1] - self._rejected_lps[-1]:.4f}")
        if self._kl_vals:
            summary_lines.append(
                f"- KL from ref: final={self._kl_vals[-1]:.4f}")
        if self._spin_lambda:
            summary_lines.append(f"- λ (spin_lambda): {self._spin_lambda}")
        summary_lines.append(f"- Steps trained: {state.global_step}")
        if self._dataset_size:
            summary_lines.append(f"- Dataset size: {self._dataset_size}")
        self.writer.add_text("spin_iterations/summary",
                             "\n".join(summary_lines), i)

        self.writer.flush()
        logger.info(
            f"SPINIterationSummaryCallback.on_train_end: iteration {i} summary flushed to TensorBoard.")

    def close(self):
        """Call after the outer SPIN loop ends."""
        logger.info(
            "SPINIterationSummaryCallback.close(): flushing and closing global TensorBoard writer.")
        self.writer.flush()
        self.writer.close()
        logger.info("Global TensorBoard writer closed.")
