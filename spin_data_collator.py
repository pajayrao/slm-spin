import logging
from typing import List, Dict, Any
import torch

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINDataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        logger.info(f"SPINDataCollator initialised — pad_token_id={self.pad_id}. "
                    f"Pads chosen/rejected input_ids, masks, and labels to batch-max length.")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
