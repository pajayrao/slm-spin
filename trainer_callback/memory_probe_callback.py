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
# MemoryProbeCallback
# ─────────────────────────────────────────────────────────────────────────────

class MemoryProbeCallback(TrainerCallback):
    """
    Logs CPU RSS and GPU allocated/reserved memory to the Python logger AND to
    TensorBoard at configurable intervals. Helps pinpoint exactly which step
    causes a memory spike.

    Metrics written (system/* namespace):
      system/gpu_alloc_mb      — GPU memory actively holding tensor data
      system/gpu_reserved_mb   — GPU memory reserved by the caching allocator
      system/cpu_rss_mb        — Process RSS (resident set size) in CPU RAM
    """

    def __init__(self, writer: SummaryWriter = None, log_every_n_steps: int = 50):
        # writer: optional SummaryWriter; if None, metrics are only printed to logger
        self.writer = writer
        self.log_every_n_steps = log_every_n_steps

    def _write(self, tag: str, step: int):
        log_memory(tag)
        if self.writer is None or not torch.cuda.is_available():
            return
        try:
            import psutil
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            self.writer.add_scalar("system/cpu_rss_mb", rss_mb, step)
        except ImportError:
            pass
        self.writer.add_scalar("system/gpu_alloc_mb",    torch.cuda.memory_allocated() / 1024 ** 2, step)
        self.writer.add_scalar("system/gpu_reserved_mb", torch.cuda.memory_reserved()  / 1024 ** 2, step)

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        self._write("train_begin", 0)

    def on_step_end(self, _args, state, _control, **_kwargs):
        # Always log the first 3 steps (warmup spikes); then throttle to every N steps
        if state.global_step < 3 or state.global_step % self.log_every_n_steps == 0:
            self._write(f"step_{state.global_step}", state.global_step)

    def on_train_end(self, _args, state, _control, **_kwargs):
        self._write("train_end", state.global_step)
