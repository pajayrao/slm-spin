import logging
from typing import List, Dict, Any
import torch

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINDataCollator:
    """Collates a list of SPINDataset items into a padded batch of tensors.

    Chosen and rejected sequences are padded independently to their respective
    batch-max lengths.  Reference log-probs are stacked into a float32 tensor
    when present.

    Example (__init__):
        Input:  tokenizer.pad_token_id=2
        Output: SPINDataCollator with self.pad_id=2; logs init message; no return value.
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        logger.info(f"SPINDataCollator initialised — pad_token_id={self.pad_id}. "
                    f"Pads chosen/rejected input_ids, masks, and labels to batch-max length.")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of dataset items into right-padded batch tensors.

        Example:
            Input:  features=[
                        {   # example 0 — shorter chosen (80 tokens), longer rejected (110 tokens)
                            "chosen_input_ids":       [1, 5, 8, ...],     # len=80
                            "chosen_attention_mask":  [1, 1, 1, ...],     # len=80
                            "chosen_labels":          [-100, -100, 8, ...], # len=80
                            "rejected_input_ids":     [1, 5, 9, ...],     # len=110
                            "rejected_attention_mask":[1, 1, 1, ...],     # len=110
                            "rejected_labels":        [-100, -100, 9, ...], # len=110
                            "ref_chosen_logp":        -12.43,
                            "ref_rejected_logp":      -18.07,
                        },
                        {   # example 1 — longer chosen (95 tokens), shorter rejected (85 tokens)
                            "chosen_input_ids":       [1, 3, 7, ...],     # len=95
                            ...
                            "ref_chosen_logp":        -9.82,
                            "ref_rejected_logp":      -14.55,
                        },
                    ]

            Output: {
                "chosen_input_ids":       tensor shape (2, 95)   # padded to max chosen len
                                          with self.pad_id=2 filling positions 80–94 of example 0
                "chosen_attention_mask":  tensor shape (2, 95)   # 0 at padding positions
                "chosen_labels":          tensor shape (2, 95)   # -100 at padding positions
                "rejected_input_ids":     tensor shape (2, 110)  # padded to max rejected len
                "rejected_attention_mask":tensor shape (2, 110)
                "rejected_labels":        tensor shape (2, 110)
                "ref_chosen_logp":        tensor([-12.43, -9.82],  dtype=torch.float32)
                "ref_rejected_logp":      tensor([-18.07, -14.55], dtype=torch.float32)
            }

        Example (no ref_logprobs):
            Input:  features without "ref_chosen_logp" / "ref_rejected_logp" keys
            Output: same dict but without the ref_chosen_logp and ref_rejected_logp entries
        """
        batch = {}
        for prefix in ["chosen", "rejected"]:
            batch[f"{prefix}_input_ids"] = pad_to_max_len(
                [f[f"{prefix}_input_ids"] for f in features], self.pad_id)
            batch[f"{prefix}_attention_mask"] = pad_to_max_len(
                [f[f"{prefix}_attention_mask"] for f in features], 0)
            batch[f"{prefix}_labels"] = pad_to_max_len(
                [f[f"{prefix}_labels"] for f in features], -100)
        if "ref_chosen_logp" in features[0]:
            batch["ref_chosen_logp"] = torch.tensor(
                [f["ref_chosen_logp"] for f in features], dtype=torch.float32)
            batch["ref_rejected_logp"] = torch.tensor(
                [f["ref_rejected_logp"] for f in features], dtype=torch.float32)

        chosen_len = batch["chosen_input_ids"].shape[1]
        rejected_len = batch["rejected_input_ids"].shape[1]
        logger.debug(
            f"SPINDataCollator: batch_size={len(features)}, "
            f"chosen_seq_len={chosen_len}, rejected_seq_len={rejected_len}, "
            f"ref_logprobs={'yes' if 'ref_chosen_logp' in batch else 'no'}."
        )
        return batch
