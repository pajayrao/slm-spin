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



class TorchProfilerCallback(TrainerCallback):
    """
    Runs the PyTorch profiler during training and exports rich diagnostics.

    Per-iteration behaviour (SPIN creates a new Trainer each iteration, so
    on_train_begin / on_train_end fire once per SPIN iteration):
      on_train_begin  — starts a fresh profiler, named by iteration + global step
      on_step_end     — advances the profiler schedule
      on_train_end    — stops profiler, exports stacks, logs key-averages table

    Outputs written to  profile_dir/run_iter_N_step_S/:
      *.pt.trace.json     TensorBoard "Trace" tab + chrome://tracing timeline
      stacks_cuda.txt     CUDA flamegraph stacks — open at speedscope.app
      stacks_cpu.txt      CPU flamegraph stacks  — open at speedscope.app

    The top-N ops table (sorted by CUDA time) is printed to the logger AND
    written as a TensorBoard text card under profiler/key_averages_iter_N.

    Args:
        cfg:           SPIN configuration.
        spin_iteration: which SPIN iteration this is — used to name the run dir
                        and to set the x-axis on the TensorBoard text card.
        tb_writer:     optional SummaryWriter to receive the key-averages text card.
                        Pass summary_cb.writer from SPINIterationSummaryCallback.
    """

    def __init__(
        self,
        cfg: SPINConfig,
        spin_iteration: int = 0,
        tb_writer: SummaryWriter = None,
    ):
        self.cfg = cfg
        self.spin_iteration = spin_iteration
        self.tb_writer = tb_writer
        self.prof = None
        self._run_dir: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.cfg.enable_profiler:
            return

        activities = []
        if self.cfg.profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if self.cfg.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        if not activities:
            logger.warning("[Profiler] enabled but no valid activities selected.")
            return

        # Include SPIN iteration in dir name so multiple iterations don't overwrite
        self._run_dir = os.path.join(
            self.cfg.profile_dir,
            f"run_iter_{self.spin_iteration}_step_{state.global_step}",
        )
        ensure_dir(self._run_dir)

        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.cfg.profile_schedule_wait,
                warmup=self.cfg.profile_schedule_warmup,
                active=self.cfg.profile_schedule_active,
                repeat=self.cfg.profile_schedule_repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self._run_dir),
            record_shapes=self.cfg.profile_record_shapes,
            profile_memory=self.cfg.profile_memory,
            with_stack=self.cfg.profile_with_stack,
            with_flops=self.cfg.profile_with_flops,
            # Records ops grouped by nn.Module name (e.g. "model.layers.0.self_attn")
            # instead of raw ATen ops. Coarser but more readable for large models.
            with_modules=self.cfg.profile_modules,
        )
        self.prof.__enter__()
        logger.info(
            f"[Profiler] Started for SPIN iter {self.spin_iteration}. "
            f"Traces → {self._run_dir}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is None:
            return

        # Check whether the scheduled on_trace_ready fired during training.
        # Must happen before __exit__ so we can distinguish "schedule fired and wrote a file"
        # from "schedule never fired and __exit__ wrote a file".
        scheduled_fired = any(
            f.endswith(".pt.trace.json")
            for f in os.listdir(self._run_dir)
            if os.path.isfile(os.path.join(self._run_dir, f))
        ) if self._run_dir and os.path.isdir(self._run_dir) else False

        # Stop the profiler to finalise all events before any export or analysis.
        # PyTorch ≥ 2.x raises RuntimeError if export_chrome_trace / export_stacks /
        # key_averages are called on a still-active profiler.
        self.prof.__exit__(None, None, None)

        if not scheduled_fired:
            fallback_path = os.path.join(self._run_dir, "trace_fallback.pt.trace.json")
            try:
                self.prof.export_chrome_trace(fallback_path)
                logger.info(f"[Profiler] Schedule never fired (training too short). Fallback trace → {fallback_path}")
            except Exception as exc:
                logger.warning(f"[Profiler] Could not export fallback trace: {exc}")

        if self.cfg.profile_export_stacks and self.cfg.profile_with_stack:
            self._export_stacks()

        if self.cfg.profile_log_top_n_ops > 0:
            self._log_key_averages()

        # Log all files written so the user knows exactly where to point TensorBoard
        if self._run_dir and os.path.isdir(self._run_dir):
            files = os.listdir(self._run_dir)
            logger.info(
                f"[Profiler] Files written to {self._run_dir}:\n"
                + "\n".join(f"  {f}" for f in files)
            )
            logger.info(
                f"[Profiler] To view in TensorBoard: "
                f"tensorboard --logdir \"{self._run_dir}\""
            )

        self.prof = None
        logger.info(f"[Profiler] Stopped for SPIN iter {self.spin_iteration}.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _export_stacks(self):
        """
        Export CPU and CUDA flamegraph stack files.
        Open at https://speedscope.app — drag and drop either .txt file.
        CUDA stacks show which kernels dominate GPU time;
        CPU stacks show Python overhead and host-side bottlenecks.
        """
        for metric, fname in [
            ("self_cuda_time_total", "stacks_cuda.txt"),
            ("self_cpu_time_total",  "stacks_cpu.txt"),
        ]:
            path = os.path.join(self._run_dir, fname)
            try:
                self.prof.export_stacks(path, metric=metric)
                logger.info(f"[Profiler] Flamegraph stacks → {path}")
            except Exception as exc:
                # CUDA stacks fail on CPU-only runs; CPU stacks fail if no CPU events were captured
                logger.debug(f"[Profiler] Skipped stacks for {metric}: {exc}")

    def _log_key_averages(self):
        """
        Print the top-N operators (by CUDA time, or CPU time for CPU-only runs)
        to the Python logger and, if a SummaryWriter was provided, as a TensorBoard
        text card under profiler/key_averages_iter_N.

        Grouping by input shape (when profile_record_shapes=True) makes the table
        distinguish e.g. matmul calls with different sequence lengths, which is
        crucial for understanding variable-length sequence overhead in SPIN.
        """
        try:
            avgs = self.prof.key_averages(
                group_by_input_shape=self.cfg.profile_record_shapes,
                group_by_stack_n=5 if self.cfg.profile_with_stack else 0,
            )

            has_cuda = any(a.self_cuda_time_total > 0 for a in avgs)
            sort_key = "self_cuda_time_total" if has_cuda else "self_cpu_time_total"
            table = avgs.table(
                sort_by=sort_key,
                row_limit=self.cfg.profile_log_top_n_ops,
            )
            logger.info(
                f"[Profiler] SPIN iter {self.spin_iteration} "
                f"top {self.cfg.profile_log_top_n_ops} ops "
                f"(sorted by {sort_key}):\n{table}"
            )

            if self.tb_writer is not None:
                self.tb_writer.add_text(
                    f"profiler/key_averages_iter_{self.spin_iteration}",
                    f"Sorted by `{sort_key}`, top {self.cfg.profile_log_top_n_ops} ops:\n\n"
                    f"```\n{table}\n```",
                    self.spin_iteration,
                )
                self.tb_writer.flush()

        except Exception as exc:
            logger.warning(f"[Profiler] Could not log key averages: {exc}")

