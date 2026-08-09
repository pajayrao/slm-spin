import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINTrainer(Trainer):
    def __init__(
        self,
        spin_lambda=0.1,
        loss_type="logistic",
        sft_alpha=0.0,
        rejected_adv_clip=None,
        kl_penalty_alpha=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spin_lambda = spin_lambda
        self.loss_type = loss_type
        # Anti likelihood-displacement regularisation — see spin_config.py for the
        # rationale behind each of these. All default to a no-op (0.0 / None) so
        # existing callers that don't pass them get the original loss behaviour.
        self.sft_alpha = sft_alpha
        self.rejected_adv_clip = rejected_adv_clip
        self.kl_penalty_alpha = kl_penalty_alpha
        # Counts optimizer steps taken by this trainer instance; used to throttle
        # verbose INFO logs so they appear on step 1 and every logging_steps thereafter.
        self._spin_step = 0
        logger.info(
            f"SPINTrainer initialised — spin_lambda={spin_lambda}, loss_type='{loss_type}', "
            f"sft_alpha={sft_alpha}, rejected_adv_clip={rejected_adv_clip}, "
            f"kl_penalty_alpha={kl_penalty_alpha}. "
            f"Margin = λ × [(π_θ(chosen)−π_ref(chosen)) − (π_θ(rejected)−π_ref(rejected))]."
        )

    def _compute_spin_loss(self, pi_chosen_logp, pi_rejected_logp, ref_chosen_logp, ref_rejected_logp):
        """Shared loss computation used by both compute_loss() and training_step().

        Combines the base SPIN margin loss with three optional anti likelihood-
        displacement regularisers (see spin_config.py):
          1. sft_alpha        — NLL anchor pulling pi_chosen_logp up directly.
          2. rejected_adv_clip — caps how much margin credit comes from cratering
             the rejected side rather than raising the chosen side.
          3. kl_penalty_alpha  — penalises a negative kl_from_ref (net regression
             vs the reference model), which is otherwise only logged, not trained on.

        All regularisers default to a no-op, so with sft_alpha=0, rejected_adv_clip=None,
        kl_penalty_alpha=0 this reproduces the original SPIN loss exactly.

        Example (loss_type="hinge", spin_lambda=0.5, sft_alpha=0.25,
                 rejected_adv_clip=5.0, kl_penalty_alpha=0.1, batch_size=1):
            Input:  pi_chosen_logp=[-1.85], pi_rejected_logp=[-24.61],
                    ref_chosen_logp=[-1.82], ref_rejected_logp=[-6.54]

            Processing:
              chosen_adv   = [-1.85 - (-1.82)] = [-0.03]
              rejected_adv = [-24.61 - (-6.54)] = [-18.07]
              rejected_adv_capped = clamp([-18.07], min=-5.0) = [-5.0]
              raw_margin   = chosen_adv - rejected_adv_capped = [-0.03 - (-5.0)] = [4.97]
              margin       = 0.5 * [4.97] = [2.485]
              base_loss    = relu(1 - 2.485).mean() = 0.0        # hinge already saturated
              sft_term     = 0.25 * -mean([-1.85]) = 0.4625       # still pulls chosen up
              kl_val       = mean(chosen_adv + rejected_adv) / 2 = (-0.03 + -18.07)/2 = -9.05
              kl_penalty   = 0.1 * relu(9.05)**2 = 8.19            # fires: kl_val is negative
              loss         = 0.0 + 0.4625 + 8.19 = 8.6525

            Output: (loss=tensor(8.6525), margin=tensor([2.485]),
                     chosen_adv=tensor([-0.03]), rejected_adv=tensor([-18.07]),
                     kl_val=tensor(-9.05))

        Compare to the unregularised loss (sft_alpha=0, rejected_adv_clip=None,
        kl_penalty_alpha=0): without clipping, raw_margin = -0.03 - (-18.07) = 18.04,
        margin = 0.5*18.04 = 9.02, base_loss = relu(1-9.02) = 0 — loss=0.0 with zero
        gradient anywhere, silently rewarding the collapse. The regularised version
        keeps a large, informative gradient via sft_term and kl_penalty even though
        the base hinge loss has saturated.
        """
        chosen_adv = pi_chosen_logp - ref_chosen_logp
        rejected_adv = pi_rejected_logp - ref_rejected_logp

        # Cap how much margin credit comes from cratering the rejected side.
        if self.rejected_adv_clip is not None:
            rejected_adv_for_margin = rejected_adv.clamp(min=-self.rejected_adv_clip)
        else:
            rejected_adv_for_margin = rejected_adv

        raw_margin = chosen_adv - rejected_adv_for_margin
        margin = self.spin_lambda * raw_margin

        if self.loss_type == "logistic":
            base_loss = F.softplus(-margin).mean()
        elif self.loss_type == "hinge":
            base_loss = F.relu(1.0 - margin).mean()
        elif self.loss_type == "correlation":
            base_loss = (1.0 - margin).mean()
        elif self.loss_type == "exponential":
            base_loss = torch.exp(-margin).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # 1. NLL anchor — reward raising pi_chosen_logp directly, not just via the margin.
        sft_term = -pi_chosen_logp.mean() * self.sft_alpha if self.sft_alpha else torch.zeros_like(base_loss)

        # 3. Penalise net regression vs the reference model (kl_from_ref < 0).
        kl_val = (chosen_adv.mean() + rejected_adv.mean()) / 2.0
        kl_penalty = (
            self.kl_penalty_alpha * F.relu(-kl_val) ** 2
            if self.kl_penalty_alpha else torch.zeros_like(base_loss)
        )

        loss = base_loss + sft_term + kl_penalty

        return loss, margin, chosen_adv, rejected_adv, kl_val, base_loss, sft_term, kl_penalty

    def _should_log_verbose(self) -> bool:
        """Return True on the very first step and every logging_steps thereafter.

        Example:
            Input:  self._spin_step=1,  self.args.logging_steps=10
            Output: True   (always log on the very first step)

            Input:  self._spin_step=10, self.args.logging_steps=10
            Output: True   (10 % 10 == 0)

            Input:  self._spin_step=5,  self.args.logging_steps=10
            Output: False  (5 % 10 != 0 and not step 1)

            Input:  self._spin_step=20, self.args.logging_steps=10
            Output: True   (20 % 10 == 0)
        """
        interval = getattr(self.args, "logging_steps", 10) or 10
        return self._spin_step == 1 or self._spin_step % interval == 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the SPIN margin loss for an evaluation batch.

        NOTE: Used by Trainer during *evaluation* (predict/evaluate), NOT during training.
        Training uses training_step() below. If you change loss logic here, mirror it there.

        Example (loss_type="logistic", batch_size=2):
            Input:  inputs={
                        "chosen_input_ids":    tensor shape (2, 128),
                        "chosen_attention_mask": tensor shape (2, 128),
                        "chosen_labels":       tensor shape (2, 128),  # prompt positions = -100
                        "rejected_input_ids":  tensor shape (2, 96),
                        "rejected_attention_mask": tensor shape (2, 96),
                        "rejected_labels":     tensor shape (2, 96),
                        "ref_chosen_logp":     tensor([-12.43, -9.82]),  # pre-scored under π_ref
                        "ref_rejected_logp":   tensor([-18.07, -14.55]),
                    }
                    model=<LlamaForCausalLM in eval mode>

            Processing:
              pi_chosen_logp   = model_sequence_logprob(chosen)   → e.g. tensor([-11.90, -9.50])
              pi_rejected_logp = model_sequence_logprob(rejected) → e.g. tensor([-17.20, -13.80])
              raw_margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
                         = [(-11.90 - -12.43) - (-17.20 - -18.07),
                            (-9.50  - -9.82)  - (-13.80 - -14.55)]
                         = [0.53 - 0.87, 0.32 - 0.75] = [-0.34, -0.43]
              margin = lambda * raw_margin  (e.g. lambda=0.1 → [-0.034, -0.043])
              loss = softplus(-margin).mean()

            Output (return_outputs=False): tensor(0.7031)  # scalar loss
            Output (return_outputs=True):  (tensor(0.7031), {
                "loss": 0.7031, "margin_mean": -0.038, "win_rate": 0.0,
                "pi_chosen_logp": -10.70, "pi_rejected_logp": -15.50, ...})
        """
        # NOTE: Used by Trainer during *evaluation* (predict/evaluate), NOT during training.
        # Training uses training_step() below. If you change loss logic here, mirror it there.

        # Step 1: Unpack input tensors from the batch dict.
        # Chosen = human reference responses; rejected = model's own previous-iteration outputs.
        # Labels have prompt positions set to -100 so only response tokens contribute to scoring.
        # Example: batch_size=2, chosen padded to 128 tokens, rejected padded to 96 tokens.
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels = inputs["chosen_labels"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels = inputs["rejected_labels"]

        bs = chosen_input_ids.size(0)
        chosen_len = chosen_input_ids.size(1)
        rejected_len = rejected_input_ids.size(1)
        logger.debug(
            f"compute_loss (eval) — batch_size={bs}, "
            f"chosen_seq_len={chosen_len}, rejected_seq_len={rejected_len}, "
            f"device={chosen_input_ids.device}"
        )

        # Step 2: Forward pass under π_θ to score human (chosen) responses.
        # This runs through the entire model with use_cache=False (inside model_sequence_logprob)
        # and returns one per-token-average log-prob scalar per sequence.
        # Called with torch.no_grad() inherited from HF Trainer's evaluation loop,
        # so no gradients are computed here — purely a scoring pass.
        # Example: chosen_input_ids shape (2,128), 70 active response tokens each
        #          pi_chosen_logp ≈ tensor([-11.90, -9.50])
        logger.debug("compute_loss: forward pass → π_θ(chosen|prompt)...")
        pi_chosen_logp = model_sequence_logprob(
            model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        logger.debug(
            f"  π_θ(chosen): mean={pi_chosen_logp.mean().item():.4f}, "
            f"min={pi_chosen_logp.min().item():.4f}, max={pi_chosen_logp.max().item():.4f}"
        )

        # Step 3: Forward pass under π_θ to score synthetic (rejected) responses.
        # Same mechanics as Step 2 but on the model's own previous-iteration generations.
        # The model tends to assign higher log-prob to these than to human responses because
        # they were sampled from it — SPIN training corrects this imbalance.
        # Example: rejected_input_ids shape (2,96), 50 active response tokens each
        #          pi_rejected_logp ≈ tensor([-17.20, -13.80])
        logger.debug("compute_loss: forward pass → π_θ(rejected|prompt)...")
        pi_rejected_logp = model_sequence_logprob(
            model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        logger.debug(
            f"  π_θ(rejected): mean={pi_rejected_logp.mean().item():.4f}, "
            f"min={pi_rejected_logp.min().item():.4f}, max={pi_rejected_logp.max().item():.4f}"
        )

        # Step 4: Load pre-computed reference log-probs from the batch.
        # These were scored offline under the frozen π_ref (compute_ref_logprobs in utils.py)
        # and stored in the dataset, so no additional forward pass is needed here.
        # Moving to the correct device handles multi-GPU or CPU-pinned-memory edge cases.
        # Example: ref_chosen_logp = tensor([-12.43, -9.82])  ← fixed, never changes during training
        #          ref_rejected_logp = tensor([-18.07, -14.55])
        ref_chosen_logp = inputs["ref_chosen_logp"].to(pi_chosen_logp.device)
        ref_rejected_logp = inputs["ref_rejected_logp"].to(
            pi_rejected_logp.device)
        logger.debug(
            f"  π_ref(chosen):   mean={ref_chosen_logp.mean().item():.4f} (pre-computed, no forward pass)"
        )
        logger.debug(
            f"  π_ref(rejected): mean={ref_rejected_logp.mean().item():.4f} (pre-computed, no forward pass)"
        )

        # Steps 5-7: advantages, margin, base loss, and anti likelihood-displacement
        # regularisation (sft_alpha / rejected_adv_clip / kl_penalty_alpha) — see
        # _compute_spin_loss() for the full breakdown. Mirrors training_step() below.
        loss, margin, chosen_adv_t, rejected_adv_t, kl_val, base_loss, sft_term, kl_penalty = (
            self._compute_spin_loss(pi_chosen_logp, pi_rejected_logp, ref_chosen_logp, ref_rejected_logp)
        )
        chosen_adv = chosen_adv_t.mean().item()
        rejected_adv = rejected_adv_t.mean().item()
        logger.debug(
            f"  chosen advantage (π_θ − π_ref):   {chosen_adv:.4f}  "
            f"(positive = model improved on human responses)"
        )
        logger.debug(
            f"  rejected advantage (π_θ − π_ref): {rejected_adv:.4f}  "
            f"(negative = model moved away from synthetic responses ✓)"
        )
        logger.debug(
            f"  scaled margin (λ={self.spin_lambda}): mean={margin.mean().item():.4f}, "
            f"win_rate={(margin > 0).float().mean().item():.3f} "
            f"({int((margin > 0).sum().item())}/{bs} examples where margin > 0)"
        )

        logger.info(
            f"compute_loss (eval) — loss={loss.item():.4f} [{self.loss_type}] "
            f"(base={base_loss.item():.4f}, sft={sft_term.item():.4f}, kl_pen={kl_penalty.item():.4f}), "
            f"margin_mean={margin.mean().item():.4f}, "
            f"win_rate={(margin > 0).float().mean().item():.3f}, "
            f"chosen_adv={chosen_adv:.4f}, rejected_adv={rejected_adv:.4f}, "
            f"batch={bs}"
        )

        # Step 8: Assemble the metrics dict for HF Trainer's evaluation logging.
        # All tensors are detached so they don't hold the computation graph in eval memory.
        # kl_from_ref ≈ mean log-ratio of π_θ to π_ref across both sides — measures overall
        # policy drift from the reference checkpoint; used for monitoring, not for the loss.
        # Example: kl_from_ref = (chosen_adv + rejected_adv) / 2 = (0.425 + 0.810) / 2 = 0.618
        #          win_rate = mean((margin > 0).float()) = 0.0  (both examples had negative margin)
        metrics = {
            "loss":              loss.detach(),
            "base_loss":         base_loss.detach(),
            "sft_term":          sft_term.detach(),
            "kl_penalty":        kl_penalty.detach(),
            "margin_mean":       margin.mean().detach(),
            "margin_std":        margin.std().detach() if margin.numel() > 1 else torch.tensor(0.0),
            "win_rate":          (margin > 0).float().mean().detach(),
            "pi_chosen_logp":    pi_chosen_logp.mean().detach(),
            "pi_rejected_logp":  pi_rejected_logp.mean().detach(),
            "ref_chosen_logp":   ref_chosen_logp.mean().detach(),
            "ref_rejected_logp": ref_rejected_logp.mean().detach(),
            "kl_from_ref":       kl_val.detach(),
        }

        return (loss, metrics) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Execute one SPIN training step: two forward passes, loss computation, and backward.

        Each call: forward chosen, forward rejected, compute SPIN margin loss, backward.
        Keeps both forward passes inside one backward to avoid double materialisation.

        Example (loss_type="logistic", batch_size=4, step=1):
            Input:  inputs={
                        "chosen_input_ids":    tensor shape (4, 150),
                        "chosen_attention_mask": tensor shape (4, 150),
                        "chosen_labels":       tensor shape (4, 150),
                        "rejected_input_ids":  tensor shape (4, 120),
                        "rejected_attention_mask": tensor shape (4, 120),
                        "rejected_labels":     tensor shape (4, 120),
                        "ref_chosen_logp":     tensor([-12.1, -10.5, -14.3, -9.8]),
                        "ref_rejected_logp":   tensor([-17.4, -15.2, -19.1, -13.6]),
                    }

            Processing:
              pi_chosen   ≈ tensor([-11.8, -10.2, -13.9, -9.5])   # model improved on chosen
              pi_rejected ≈ tensor([-16.9, -14.8, -18.5, -13.2])  # model moved away from rejected
              chosen_adv  = pi_chosen  - ref_chosen    ≈ [+0.3, +0.3, +0.4, +0.3]
              rejected_adv= pi_rejected- ref_rejected  ≈ [+0.5, +0.4, +0.6, +0.4]
              raw_margin  = chosen_adv - rejected_adv  ≈ [-0.2, -0.1, -0.2, -0.1]  (pre-λ)
              margin      = lambda * raw_margin        ≈ [-0.02, -0.01, -0.02, -0.01]
              loss        = softplus(-margin).mean()   ≈ 0.6981
              loss.backward() called internally

            Output: tensor(0.6981)  # detached loss scalar returned to HF Trainer
            Side effects:
              - gradients accumulated on LoRA adapter parameters
              - log_data dict sent to self.log() for TensorBoard/WandB
              - throttled INFO log printed on step 1 and every logging_steps thereafter
        """
        # Each call: forward chosen, forward rejected, compute SPIN margin loss, backward.
        # Keeps both forward passes inside one backward to avoid double materialisation.

        # Step 1: Set model to training mode and move all inputs to the correct device.
        # model.train() re-enables dropout and batch-norm updates — critical if any eval
        # forward pass called model.eval() before this step.
        # _prepare_inputs handles .to(device) and mixed-precision dtype casting for every
        # tensor in inputs, so all downstream code can assume tensors are on-device.
        # Example: inputs tensors arrive on CPU → after _prepare_inputs all on cuda:0,
        #          chosen_input_ids.device == rejected_input_ids.device == cuda:0
        self._spin_step += 1
        model.train()
        inputs = self._prepare_inputs(inputs)

        bs = inputs["chosen_input_ids"].size(0)
        chosen_len = inputs["chosen_input_ids"].size(1)
        rejected_len = inputs["rejected_input_ids"].size(1)
        dev = next(model.parameters()).device

        logger.debug(
            f"training_step #{self._spin_step} — batch_size={bs}, "
            f"chosen_len={chosen_len}, rejected_len={rejected_len}, device={dev}"
        )

        # Step 2: Load pre-computed reference model log-probs from the batch.
        # These were scored under the frozen π_ref before training started (compute_ref_logprobs
        # in utils.py) and stored in the dataset, so no second model forward pass is needed here.
        # Moving to dev ensures tensor ops in later steps don't hit device-mismatch errors.
        # Example: ref_chosen_logp = tensor([-12.1, -10.5, -14.3, -9.8], device='cuda:0')
        #          ref_rejected_logp = tensor([-17.4, -15.2, -19.1, -13.6], device='cuda:0')
        #          Both fixed constants — they do NOT change during training.
        ref_chosen_logp = inputs["ref_chosen_logp"].to(dev)
        ref_rejected_logp = inputs["ref_rejected_logp"].to(dev)

        logger.debug(
            f"  π_ref(chosen)   — mean={ref_chosen_logp.mean().item():.4f}, "
            f"min={ref_chosen_logp.min().item():.4f}, max={ref_chosen_logp.max().item():.4f}"
        )
        logger.debug(
            f"  π_ref(rejected) — mean={ref_rejected_logp.mean().item():.4f}, "
            f"min={ref_rejected_logp.min().item():.4f}, max={ref_rejected_logp.max().item():.4f}"
        )

        # Step 3: Forward pass 1 — score human (chosen) responses under the current model π_θ.
        # model_sequence_logprob runs a full forward pass (no KV-cache) and returns one
        # per-token-average log-prob scalar per sequence in the batch.
        # chosen_labels has prompt positions set to -100 so only response tokens are scored.
        # Gradients flow through this pass because model.train() is active (no torch.no_grad).
        # Example: chosen_input_ids shape (4,150), chosen_labels has 80 response tokens per seq
        #          pi_chosen ≈ tensor([-11.8, -10.2, -13.9, -9.5])  — one scalar per example
        #          More negative = model is less confident about those human response tokens.
        logger.debug("  Forward pass 1/2: π_θ(chosen|prompt)...")
        pi_chosen = model_sequence_logprob(
            model, inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"], inputs["chosen_labels"],
        )
        logger.debug(
            f"  π_θ(chosen)  — mean={pi_chosen.mean().item():.4f}, "
            f"min={pi_chosen.min().item():.4f}, max={pi_chosen.max().item():.4f}"
        )

        # Step 4: Forward pass 2 — score synthetic (rejected) responses under π_θ.
        # Same mechanics as Step 3 but on the model's own previous-iteration generations.
        # Both forward passes stay inside a single backward call (loss.backward() below),
        # so their computation graphs are live simultaneously — no double materialisation.
        # Example: rejected_input_ids shape (4,120), rejected_labels has 60 response tokens
        #          pi_rejected ≈ tensor([-16.9, -14.8, -18.5, -13.2])
        #          These are the model's own old outputs, so it tends to assign higher
        #          log-prob to them than to human responses — the SPIN loss corrects this.
        logger.debug("  Forward pass 2/2: π_θ(rejected|prompt)...")
        pi_rejected = model_sequence_logprob(
            model, inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"], inputs["rejected_labels"],
        )
        logger.debug(
            f"  π_θ(rejected) — mean={pi_rejected.mean().item():.4f}, "
            f"min={pi_rejected.min().item():.4f}, max={pi_rejected.max().item():.4f}"
        )

        # Steps 5-7: advantages, margin, base loss, and anti likelihood-displacement
        # regularisation (sft_alpha / rejected_adv_clip / kl_penalty_alpha). See
        # _compute_spin_loss() for the full breakdown and worked example — this is the
        # single most important change from the original SPIN loss: without it, the
        # optimizer can satisfy the margin purely by cratering pi_rejected far below
        # pi_ref(rejected) instead of raising pi_chosen, which was observed in practice
        # (rejected_adv reaching -18 while chosen_adv stayed near 0). Mirrors compute_loss() above.
        loss, margin, chosen_adv, rejected_adv, kl_val, base_loss, sft_term, kl_penalty = (
            self._compute_spin_loss(pi_chosen, pi_rejected, ref_chosen_logp, ref_rejected_logp)
        )
        logger.debug(
            f"  chosen advantage (π_θ−π_ref): mean={chosen_adv.mean().item():.4f} "
            f"(want > 0: model getting better at human responses)"
        )
        logger.debug(
            f"  rejected advantage (π_θ−π_ref): mean={rejected_adv.mean().item():.4f} "
            f"(want < 0: model moving away from its own old generations)"
        )
        logger.debug(
            f"  scaled margin (λ={self.spin_lambda}): mean={margin.mean().item():.4f}, "
            f"base_loss={base_loss.item():.4f}, sft_term={sft_term.item():.4f}, "
            f"kl_penalty={kl_penalty.item():.4f}"
        )

        # Step 8: Backward pass — differentiate the loss through both forward passes.
        # The loss is divided by gradient_accumulation_steps BEFORE backward so the
        # accumulated gradient over GA micro-batches equals the mean-loss gradient,
        # matching stock HF Trainer behaviour. Without this division (the original
        # bug here) accumulated gradients were GA× too large, which made
        # max_grad_norm=1.0 clip on essentially every optimizer step and destroyed
        # the configured LR semantics.
        # For LoRA the gradient lands only on the low-rank adapter matrices, not the
        # frozen base weights; the optimizer consumes .grad in its step() call
        # (handled by HF Trainer after this method returns).
        ga_steps = max(1, self.args.gradient_accumulation_steps)
        logger.debug(
            f"  loss ({self.loss_type}): {loss.item():.6f}. Running backward "
            f"(scaled by 1/{ga_steps} for gradient accumulation)...")
        (loss / ga_steps).backward()
        logger.debug("  Backward pass complete. Gradients accumulated.")

        # Gradient global norm — computed immediately after backward(), while gradients
        # are still alive on the parameters (HF Trainer calls model.zero_grad() BEFORE
        # firing on_step_end, so a callback would always read 0/None). Only computed on
        # optimizer-step boundaries (see Step 10) — mid-accumulation norms conflate
        # partial sums, and the stack+norm op costs a GPU sync per call.
        at_optim_boundary = (self._spin_step % ga_steps == 0)
        if at_optim_boundary:
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            grad_global_norm = (
                torch.stack([g.float().norm().pow(2) for g in grads]).sum().item() ** 0.5
                if grads else 0.0
            )
        else:
            grad_global_norm = None

        # Step 9: Detach all tensors to plain Python floats before building the metrics dict.
        # .detach() breaks the autograd graph so these scalars don't keep the computation
        # graph alive in memory after this step.
        # kl_from_ref ≈ average log-ratio of π_θ to π_ref across both sides — a rough
        # measure of how far the model has drifted from the reference checkpoint overall.
        # win_rate = fraction of examples where margin > 0 (model improving on chosen > rejected);
        # ideally increases toward 1.0 as training progresses.
        # Example: margin=[-0.02, +0.05, -0.01, +0.08] (batch=4)
        #          win_rate = 2/4 = 0.50  (2 examples had positive margin)
        #          kl_from_ref = (chosen_adv_mean + rejected_adv_mean) / 2
        #                      = (0.325 + 0.425) / 2 = 0.375  (model drifted +0.375 nats from ref)
        n_wins = int((margin > 0).sum().item())
        win_rate = n_wins / bs if bs > 0 else 0.0
        margin_mean = margin.mean().detach().item()
        margin_std = margin.std().detach().item() if margin.numel() > 1 else 0.0
        loss_val = loss.detach().item()
        base_loss_val = base_loss.detach().item()
        sft_term_val = sft_term.detach().item()
        kl_penalty_val = kl_penalty.detach().item()
        pi_chosen_mean = pi_chosen.mean().detach().item()
        pi_rej_mean = pi_rejected.mean().detach().item()
        ref_ch_mean = ref_chosen_logp.mean().item()
        ref_rej_mean = ref_rejected_logp.mean().item()
        chosen_adv_mean = chosen_adv.mean().detach().item()
        rejected_adv_mean = rejected_adv.mean().detach().item()
        kl_val_scalar = kl_val.detach().item()

        # Step 10: Build and emit the metrics dict for TensorBoard / WandB logging.
        # Emitted only on optimizer-step boundaries: Trainer.log() fires callbacks
        # immediately on every call, so logging per micro-step (the original
        # behaviour) wrote each TensorBoard tag 32× at the same global_step, spammed
        # the console via ProgressCallback, and grew state.log_history by ~50k
        # entries per iteration for no information gain.
        if at_optim_boundary:
            log_data = {
                "loss":              loss_val,
                "base_loss":         base_loss_val,
                "sft_term":          sft_term_val,
                "kl_penalty":        kl_penalty_val,
                "margin_mean":       margin_mean,
                "margin_std":        margin_std,
                "win_rate":          win_rate,
                "pi_chosen_logp":    pi_chosen_mean,
                "pi_rejected_logp":  pi_rej_mean,
                "ref_chosen_logp":   ref_ch_mean,
                "ref_rejected_logp": ref_rej_mean,
                "kl_from_ref":       kl_val_scalar,
                "spin_lambda":       self.spin_lambda,
                "grad_global_norm":  grad_global_norm,
            }
            if self.optimizer is not None:
                log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            self.log(log_data)

        # Throttled INFO log: step 1 + every logging_steps gives a human-readable summary.
        if self._should_log_verbose():
            lr_str = f", lr={self.optimizer.param_groups[0]['lr']:.2e}" if self.optimizer else ""
            logger.info(
                f"training_step #{self._spin_step}{lr_str} — "
                f"loss={loss_val:.4f} (base={base_loss_val:.4f}, sft={sft_term_val:.4f}, "
                f"kl_pen={kl_penalty_val:.4f}), margin={margin_mean:.4f}±{margin_std:.4f}, "
                f"win_rate={win_rate:.3f} ({n_wins}/{bs} positive), "
                f"π_θ(chosen)={pi_chosen_mean:.4f}, π_θ(rejected)={pi_rej_mean:.4f}, "
                f"π_ref(chosen)={ref_ch_mean:.4f}, π_ref(rejected)={ref_rej_mean:.4f}, "
                f"chosen_adv={chosen_adv_mean:.4f}, "
                f"rejected_adv={rejected_adv_mean:.4f}, "
                f"kl_from_ref={kl_val_scalar:.4f}"
            )
        else:
            logger.debug(
                f"training_step #{self._spin_step} — "
                f"loss={loss_val:.4f}, margin={margin_mean:.4f}, win_rate={win_rate:.3f}"
            )

        return loss.detach() / self.args.gradient_accumulation_steps


class RMSPropSPINTrainer(SPINTrainer):
    def create_optimizer(self):
        """Build an RMSprop optimizer with separate weight-decay parameter groups.

        Called lazily by the HuggingFace Trainer before the first training step.
        Splits parameters into two groups: weight-decay params (linear/embedding weights)
        and no-decay params (biases and layer norms), matching standard practice.

        Example:
            Input:  self.model=<PeftModel with 8.4M LoRA params (requires_grad=True)>,
                    self.args.learning_rate=5e-5,
                    self.args.weight_decay=0.01

            Processing:
              decay_parameters  = names of weight matrices (e.g. "lora_A.weight", "lora_B.weight")
              no_decay_parameters = bias and norm params (e.g. "bias", "layer_norm.weight")
              optimizer_grouped_parameters = [
                  {"params": <8.2M weight params>, "weight_decay": 0.01},
                  {"params": <0.2M bias/norm params>, "weight_decay": 0.0},
              ]

            Output: torch.optim.RMSprop(
                        optimizer_grouped_parameters,
                        lr=5e-5, alpha=0.99, eps=1e-8,
                        momentum=0.0, centered=False, foreach=True
                    )
                    Stored at self.optimizer and also returned.

        Note: foreach=True uses vectorized parameter updates for better GPU throughput
        compared to the default per-parameter loop.
        """
        if self.optimizer is None:
            logger.info(
                "RMSPropSPINTrainer.create_optimizer() — building RMSprop optimiser...")
            decay_parameters = self.get_decay_parameter_names(self.model)
            decay_params = [
                p for n, p in self.model.named_parameters()
                if n in decay_parameters and p.requires_grad
            ]
            no_decay_params = [
                p for n, p in self.model.named_parameters()
                if n not in decay_parameters and p.requires_grad
            ]
            logger.info(f"  Parameter groups: {len(decay_params)} weight-decay params "
                        f"(wd={self.args.weight_decay}), "
                        f"{len(no_decay_params)} no-decay params (biases/norms, wd=0.0).")
            optimizer_grouped_parameters = [
                {"params": decay_params,    "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            self.optimizer = torch.optim.RMSprop(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                alpha=0.99,
                eps=1e-8,
                momentum=0.0,
                centered=False,
                foreach=True,
            )
            total_trainable = sum(p.numel()
                                  for p in decay_params + no_decay_params)
            logger.info(f"  RMSprop created: lr={self.args.learning_rate:.2e}, alpha=0.99, eps=1e-8, "
                        f"foreach=True. Total trainable params: {total_trainable:,}.")
        return self.optimizer
