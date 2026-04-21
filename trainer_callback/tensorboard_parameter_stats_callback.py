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
# TensorBoardParameterStatsCallback
# ─────────────────────────────────────────────────────────────────────────────

class TensorBoardParameterStatsCallback(TrainerCallback):
    """
    Snapshots parameter statistics at the start and end of each SPIN iteration
    and writes them to a per-iteration SummaryWriter. Tracks how far each layer
    has moved from its initial weights.

    NOTE: Previously this class had a critical indentation bug — all methods
    were accidentally defined at module level and never bound to the class.
    This is the corrected version.
    """

    def __init__(self, cfg: SPINConfig, run_name: str):
        self.cfg = cfg
        self.run_name = run_name
        self.writer: SummaryWriter = None
        self.initial_params: dict = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        log_dir = os.path.join(self.cfg.output_dir, "param_stats", self.run_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        if model is None:
            return

        # Snapshot initial weights (step 0) and write baseline histograms/scalars
        logged = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if logged >= self.cfg.parameter_log_max_tensors:
                break

            w = param.detach().float().cpu()
            self.initial_params[name] = w.clone()
            tag = name.replace(".", "/")

            if self.cfg.log_parameter_histograms:
                self.writer.add_histogram(f"parameters/{tag}", w, 0)
            if self.cfg.log_parameter_scalars:
                self.writer.add_scalar(f"parameters/mean/{tag}",   w.mean().item(),      0)
                self.writer.add_scalar(f"parameters/std/{tag}",    w.std().item(),       0)
                self.writer.add_scalar(f"parameters/norm/{tag}",   w.norm().item(),      0)
                self.writer.add_scalar(f"parameters/absmax/{tag}", w.abs().max().item(), 0)
            logged += 1

        self.writer.flush()

    def _should_log(self, step: int) -> bool:
        return step == 0 or step % self.cfg.parameter_log_interval == 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.writer is None or model is None:
            return
        if not self._should_log(state.global_step):
            return

        logged = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if logged >= self.cfg.parameter_log_max_tensors:
                break

            p   = param.detach().float().cpu()
            tag = name.replace(".", "/")

            if self.cfg.log_parameter_histograms:
                self.writer.add_histogram(f"parameters/{tag}", p, state.global_step)
            if self.cfg.log_parameter_scalars:
                self.writer.add_scalar(f"parameters/mean/{tag}",   p.mean().item(),      state.global_step)
                self.writer.add_scalar(f"parameters/std/{tag}",    p.std().item(),       state.global_step)
                self.writer.add_scalar(f"parameters/norm/{tag}",   p.norm().item(),      state.global_step)
                self.writer.add_scalar(f"parameters/absmax/{tag}", p.abs().max().item(), state.global_step)

            # Delta from initial weights: shows which layers are changing most
            if name in self.initial_params:
                delta = p - self.initial_params[name]
                self.writer.add_scalar(f"parameter_delta/norm/{tag}", delta.norm().item(), state.global_step)
                self.writer.add_scalar(f"parameter_delta/mean/{tag}", delta.mean().item(), state.global_step)
                self.writer.add_scalar(f"parameter_delta/std/{tag}",  delta.std().item(),  state.global_step)

            # Gradient stats
            if self.cfg.log_gradient_histograms and param.grad is not None:
                g = param.grad.detach().float().cpu()
                self.writer.add_histogram(f"gradients/{tag}",        g,                   state.global_step)
                self.writer.add_scalar(f"gradients/norm/{tag}",   g.norm().item(),      state.global_step)
                self.writer.add_scalar(f"gradients/absmax/{tag}", g.abs().max().item(), state.global_step)

            logged += 1

        self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

