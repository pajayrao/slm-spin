import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINTrainer(Trainer):
    def __init__(self, spin_lambda=0.1, loss_type="logistic", **kwargs):
        super().__init__(**kwargs)
        self.spin_lambda = spin_lambda
        self.loss_type = loss_type
        # Counts optimizer steps taken by this trainer instance; used to throttle
        # verbose INFO logs so they appear on step 1 and every logging_steps thereafter.
        self._spin_step = 0
        logger.info(
            f"SPINTrainer initialised — spin_lambda={spin_lambda}, loss_type='{loss_type}'. "
            f"Margin = λ × [(π_θ(chosen)−π_ref(chosen)) − (π_θ(rejected)−π_ref(rejected))]."
        )

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

        # π_θ(chosen|prompt): how likely the model under training assigns to human responses.
        logger.debug("compute_loss: forward pass → π_θ(chosen|prompt)...")
        pi_chosen_logp = model_sequence_logprob(
            model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        logger.debug(
            f"  π_θ(chosen): mean={pi_chosen_logp.mean().item():.4f}, "
            f"min={pi_chosen_logp.min().item():.4f}, max={pi_chosen_logp.max().item():.4f}"
        )

        # π_θ(rejected|prompt): how likely the model assigns to its own old (synthetic) responses.
        logger.debug("compute_loss: forward pass → π_θ(rejected|prompt)...")
        pi_rejected_logp = model_sequence_logprob(
            model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        logger.debug(
            f"  π_θ(rejected): mean={pi_rejected_logp.mean().item():.4f}, "
            f"min={pi_rejected_logp.min().item():.4f}, max={pi_rejected_logp.max().item():.4f}"
        )

        # Pre-scored under the frozen π_ref; loaded from the batch — no forward pass needed.
        ref_chosen_logp = inputs["ref_chosen_logp"].to(pi_chosen_logp.device)
        ref_rejected_logp = inputs["ref_rejected_logp"].to(
            pi_rejected_logp.device)
        logger.debug(
            f"  π_ref(chosen):   mean={ref_chosen_logp.mean().item():.4f} (pre-computed, no forward pass)"
        )
        logger.debug(
            f"  π_ref(rejected): mean={ref_rejected_logp.mean().item():.4f} (pre-computed, no forward pass)"
        )

        # Advantage per side: how much the current model has shifted vs the reference.
        chosen_adv = (pi_chosen_logp - ref_chosen_logp).mean().item()
        rejected_adv = (pi_rejected_logp - ref_rejected_logp).mean().item()
        logger.debug(
            f"  chosen advantage (π_θ − π_ref):   {chosen_adv:.4f}  "
            f"(positive = model improved on human responses)"
        )
        logger.debug(
            f"  rejected advantage (π_θ − π_ref): {rejected_adv:.4f}  "
            f"(negative = model moved away from synthetic responses ✓)"
        )

        # SPIN margin: positive → model now prefers human responses more than π_ref did.
        raw_margin = (pi_chosen_logp - ref_chosen_logp) - \
            (pi_rejected_logp - ref_rejected_logp)
        margin = self.spin_lambda * raw_margin
        logger.debug(
            f"  raw margin (before λ): mean={raw_margin.mean().item():.4f}, "
            f"std={raw_margin.std().item():.4f}, "
            f"min={raw_margin.min().item():.4f}, max={raw_margin.max().item():.4f}"
        )
        logger.debug(
            f"  scaled margin (λ={self.spin_lambda}): mean={margin.mean().item():.4f}, "
            f"win_rate={(margin > 0).float().mean().item():.3f} "
            f"({int((margin > 0).sum().item())}/{bs} examples where margin > 0)"
        )

        if self.loss_type == "logistic":
            loss = F.softplus(-margin).mean()
        elif self.loss_type == "hinge":
            loss = F.relu(1.0 - margin).mean()
        elif self.loss_type == "correlation":
            loss = (1.0 - margin).mean()
        elif self.loss_type == "exponential":
            loss = torch.exp(-margin).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        logger.info(
            f"compute_loss (eval) — loss={loss.item():.4f} [{self.loss_type}], "
            f"margin_mean={margin.mean().item():.4f}, "
            f"win_rate={(margin > 0).float().mean().item():.3f}, "
            f"chosen_adv={chosen_adv:.4f}, rejected_adv={rejected_adv:.4f}, "
            f"batch={bs}"
        )

        metrics = {
            "loss":              loss.detach(),
            "margin_mean":       margin.mean().detach(),
            "margin_std":        margin.std().detach() if margin.numel() > 1 else torch.tensor(0.0),
            "win_rate":          (margin > 0).float().mean().detach(),
            "pi_chosen_logp":    pi_chosen_logp.mean().detach(),
            "pi_rejected_logp":  pi_rejected_logp.mean().detach(),
            "ref_chosen_logp":   ref_chosen_logp.mean().detach(),
            "ref_rejected_logp": ref_rejected_logp.mean().detach(),
            "kl_from_ref":       (
                (pi_chosen_logp - ref_chosen_logp).mean() +
                (pi_rejected_logp - ref_rejected_logp).mean()
            ).detach() / 2.0,
        }

        return (loss, metrics) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch):
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

        # Forward pass: human response under π_θ.
        logger.debug("  Forward pass 1/2: π_θ(chosen|prompt)...")
        pi_chosen = model_sequence_logprob(
            model, inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"], inputs["chosen_labels"],
        )
        logger.debug(
            f"  π_θ(chosen)  — mean={pi_chosen.mean().item():.4f}, "
            f"min={pi_chosen.min().item():.4f}, max={pi_chosen.max().item():.4f}"
        )

        # Forward pass: synthetic (rejected) response under π_θ.
        logger.debug("  Forward pass 2/2: π_θ(rejected|prompt)...")
        pi_rejected = model_sequence_logprob(
            model, inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"], inputs["rejected_labels"],
        )
        logger.debug(
            f"  π_θ(rejected) — mean={pi_rejected.mean().item():.4f}, "
            f"min={pi_rejected.min().item():.4f}, max={pi_rejected.max().item():.4f}"
        )

        # Advantage per side shows which direction the model is moving relative to π_ref.
        chosen_adv = (pi_chosen - ref_chosen_logp)
        rejected_adv = (pi_rejected - ref_rejected_logp)
        logger.debug(
            f"  chosen advantage (π_θ−π_ref): mean={chosen_adv.mean().item():.4f} "
            f"(want > 0: model getting better at human responses)"
        )
        logger.debug(
            f"  rejected advantage (π_θ−π_ref): mean={rejected_adv.mean().item():.4f} "
            f"(want < 0: model moving away from its own old generations)"
        )

        raw_margin = chosen_adv - rejected_adv
        margin = self.spin_lambda * raw_margin
        logger.debug(
            f"  raw margin (before λ): mean={raw_margin.mean().item():.4f}, "
            f"std={raw_margin.std().item():.4f}, "
            f"min={raw_margin.min().item():.4f}, max={raw_margin.max().item():.4f}"
        )

        if self.loss_type == "logistic":
            loss = F.softplus(-margin).mean()
        elif self.loss_type == "hinge":
            loss = F.relu(1.0 - margin).mean()
        elif self.loss_type == "correlation":
            loss = (1.0 - margin).mean()
        elif self.loss_type == "exponential":
            loss = torch.exp(-margin).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        logger.debug(
            f"  loss ({self.loss_type}): {loss.item():.6f}. Running backward...")
        loss.backward()
        logger.debug("  Backward pass complete. Gradients accumulated.")

        # Detach all scalars before building log_data to avoid holding the graph.
        win_rate = (margin > 0).float().mean().detach().item()
        margin_mean = margin.mean().detach().item()
        margin_std = margin.std().detach().item() if margin.numel() > 1 else 0.0
        loss_val = loss.detach().item()
        pi_chosen_mean = pi_chosen.mean().detach().item()
        pi_rej_mean = pi_rejected.mean().detach().item()
        ref_ch_mean = ref_chosen_logp.mean().item()
        ref_rej_mean = ref_rejected_logp.mean().item()
        chosen_adv_mean = chosen_adv.mean().detach().item()
        rejected_adv_mean = rejected_adv.mean().detach().item()
        kl_val = (chosen_adv_mean + rejected_adv_mean) / 2.0

        log_data = {
            "loss":              loss_val,
            "margin_mean":       margin_mean,
            "margin_std":        margin_std,
            "win_rate":          win_rate,
            "pi_chosen_logp":    pi_chosen_mean,
            "pi_rejected_logp":  pi_rej_mean,
            "ref_chosen_logp":   ref_ch_mean,
            "ref_rejected_logp": ref_rej_mean,
            "kl_from_ref":       kl_val,
            "spin_lambda":       self.spin_lambda,
        }
        if self.optimizer is not None:
            log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.log(log_data)

        # Throttled INFO log: step 1 + every logging_steps gives a human-readable summary.
        if self._should_log_verbose():
            lr_str = f", lr={self.optimizer.param_groups[0]['lr']:.2e}" if self.optimizer else ""
            logger.info(
                f"training_step #{self._spin_step}{lr_str} — "
                f"loss={loss_val:.4f}, margin={margin_mean:.4f}±{margin_std:.4f}, "
                f"win_rate={win_rate:.3f} ({int(win_rate * bs)}/{bs} positive), "
                f"π_θ(chosen)={pi_chosen_mean:.4f}, π_θ(rejected)={pi_rej_mean:.4f}, "
                f"π_ref(chosen)={ref_ch_mean:.4f}, π_ref(rejected)={ref_rej_mean:.4f}, "
                f"chosen_adv={chosen_adv_mean:.4f}, "
                f"rejected_adv={rejected_adv_mean:.4f}, "
                f"kl_from_ref={kl_val:.4f}"
            )
        else:
            logger.debug(
                f"training_step #{self._spin_step} — "
                f"loss={loss_val:.4f}, margin={margin_mean:.4f}, win_rate={win_rate:.3f}"
            )

        return loss.detach()


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
