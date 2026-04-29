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
        # spin_lambda (λ): scales the margin. Higher λ means a steeper gradient
        # signal — the model is pushed harder to separate chosen from rejected.
        self.spin_lambda = spin_lambda
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # NOTE: This path is used by Trainer during *evaluation* (predict/evaluate).
        # Training uses training_step() below, which splits the backward pass to
        # save GPU activation memory. If you change loss logic here, mirror it there.

        chosen_input_ids      = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels         = inputs["chosen_labels"]

        rejected_input_ids      = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels         = inputs["rejected_labels"]

        # π_θ(chosen | prompt): log-prob of the human response under the model being trained.
        # Higher is better — the model should assign high probability to human responses.
        pi_chosen_logp = model_sequence_logprob(model, chosen_input_ids, chosen_attention_mask, chosen_labels)

        # π_θ(rejected | prompt): log-prob of the synthetic (prev-model) response under the model being trained.
        # Lower is better — the model should assign low probability to its own old generations.
        pi_rejected_logp = model_sequence_logprob(model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        # Pre-computed log-probs from the frozen reference model (previous SPIN iteration).
        # These are scalars loaded from the batch — no GPU forward pass needed here.
        ref_chosen_logp  = inputs["ref_chosen_logp"].to(pi_chosen_logp.device)
        ref_rejected_logp = inputs["ref_rejected_logp"].to(pi_rejected_logp.device)

        # SPIN margin = λ × [(π_θ(chosen) − π_ref(chosen)) − (π_θ(rejected) − π_ref(rejected))]
        # Intuitively: how much more has the current model improved on chosen vs rejected
        # compared to the reference model. A positive margin means training is working.
        margin = self.spin_lambda * (
            (pi_chosen_logp - ref_chosen_logp) - (pi_rejected_logp - ref_rejected_logp)
        )

        # Map margin to a scalar loss. All variants are minimised when margin > 0.
        if self.loss_type == "logistic":
            # softplus(-margin) ≈ log(1 + e^{-margin}): smooth, never zero, strong gradient for negative margins.
            loss = F.softplus(-margin).mean()
        elif self.loss_type == "hinge":
            # relu(1 − margin): zero loss once margin exceeds 1, hard margin boundary.
            loss = F.relu(1.0 - margin).mean()
        elif self.loss_type == "correlation":
            # (1 − margin): linear penalty, no saturation — gradient is constant regardless of margin.
            loss = (1.0 - margin).mean()
        elif self.loss_type == "exponential":
            # exp(−margin): very aggressive on negative margins, can be unstable.
            loss = torch.exp(-margin).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

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
        # Custom training step that avoids holding chosen AND rejected activations
        # in GPU memory simultaneously. Standard autograd would keep both alive
        # until loss.backward() — on a small GPU this alone causes OOM.
        # Instead we use three passes:
        #   Pass 1 (no_grad): compute the margin scalar needed to derive gradient scales.
        #   Pass 2 (grad):    chosen forward + backward → activations freed.
        #   Pass 3 (grad):    rejected forward + backward → activations freed.
        # Peak activation memory = max(chosen, rejected) instead of chosen + rejected.

        model.train()
        inputs = self._prepare_inputs(inputs)

        dev = next(model.parameters()).device
        # Pre-computed reference log-probs (frozen prev-iteration model, computed before training).
        ref_chosen_logp  = inputs["ref_chosen_logp"].to(dev)
        ref_rejected_logp = inputs["ref_rejected_logp"].to(dev)

        # ── Pass 1: no_grad — compute margin and gradient scales ─────────────
        # We need both logprobs to compute the margin before we can backward either,
        # so we run both cheaply without storing activations, then derive the
        # per-example gradient of loss w.r.t. each logprob via the chain rule.
        with torch.no_grad():
            pi_c_det = model_sequence_logprob(
                model, inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"], inputs["chosen_labels"],
            )
            pi_r_det = model_sequence_logprob(
                model, inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"], inputs["rejected_labels"],
            )

            margin = self.spin_lambda * ((pi_c_det - ref_chosen_logp) - (pi_r_det - ref_rejected_logp))
            N = margin.numel()  # number of examples in the batch

            # Compute ∂loss/∂margin per element (depends on loss type).
            if self.loss_type == "logistic":
                loss = F.softplus(-margin).mean()
                g = -torch.sigmoid(-margin) / N   # derivative of softplus(-x) is -sigmoid(-x)
            elif self.loss_type == "hinge":
                loss = F.relu(1.0 - margin).mean()
                g = -(margin < 1).float() / N     # subgradient: −1 where margin < 1, else 0
            elif self.loss_type == "correlation":
                loss = (1.0 - margin).mean()
                g = -torch.ones_like(margin) / N  # constant gradient, always −1/N
            elif self.loss_type == "exponential":
                loss = torch.exp(-margin).mean()
                g = -torch.exp(-margin) / N       # derivative of exp(−x) is −exp(−x)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Chain rule through margin = λ × (pi_chosen − pi_rejected − constants):
            #   ∂loss/∂pi_chosen  =  ∂loss/∂margin × ∂margin/∂pi_chosen  =  g × λ
            #   ∂loss/∂pi_rejected = ∂loss/∂margin × ∂margin/∂pi_rejected = g × (−λ)
            grad_chosen   = ( g * self.spin_lambda)
            grad_rejected = (-g * self.spin_lambda)

        # ── Pass 2: chosen forward + backward ───────────────────────────────
        # Run a full forward pass with grad tracking for the chosen sequence.
        # pi_chosen.backward(grad_chosen) injects our pre-computed gradient scale
        # instead of backpropping through the loss again — this is mathematically
        # identical to the standard path but allows us to free these activations
        # before allocating the rejected activations in pass 3.
        pi_chosen = model_sequence_logprob(
            model, inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"], inputs["chosen_labels"],
        )
        pi_chosen.backward(grad_chosen)   # gradients accumulate into model .grad buffers

        # ── Pass 3: rejected forward + backward ─────────────────────────────
        # Chosen activations have been released by this point (backward freed them).
        # Only rejected activations are live, so peak memory is halved vs standard.
        pi_rejected = model_sequence_logprob(
            model, inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"], inputs["rejected_labels"],
        )
        pi_rejected.backward(grad_rejected)  # accumulates into same .grad buffers (adds, not replaces)

        log_data = {
            "loss":              loss.item(),
            "margin_mean":       margin.mean().item(),
            "margin_std":        margin.std().item() if margin.numel() > 1 else 0.0,
            "win_rate":          (margin > 0).float().mean().item(),
            "pi_chosen_logp":    pi_c_det.mean().item(),
            "pi_rejected_logp":  pi_r_det.mean().item(),
            "ref_chosen_logp":   ref_chosen_logp.mean().item(),
            "ref_rejected_logp": ref_rejected_logp.mean().item(),
            "kl_from_ref":       (
                (pi_c_det - ref_chosen_logp).mean().item() +
                (pi_r_det - ref_rejected_logp).mean().item()
            ) / 2.0,
            "spin_lambda":       self.spin_lambda,
        }
        if self.optimizer is not None:
            # Current LR from scheduler (useful for diagnosing warmup / decay behaviour).
            log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.log(log_data)

        return loss.detach()


class RMSPropSPINTrainer(SPINTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            # Split parameters into two groups: those that receive weight decay
            # (typically weight matrices) and those that don't (biases, norms).
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,  # no decay on biases / layer norms
                },
            ]
            self.optimizer = torch.optim.RMSprop(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                alpha=0.99,   # smoothing factor for the running squared-gradient average
                eps=1e-8,     # added to denominator for numerical stability
                momentum=0.0,
                centered=False,
                foreach=True,   # fuses the update across all params into fewer CUDA kernels; ~10% faster per step
            )
        return self.optimizer
