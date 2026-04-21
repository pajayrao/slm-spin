import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import get_last_checkpoint

from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINDataCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for prefix in ["chosen", "rejected"]:
            batch[f"{prefix}_input_ids"] = pad_to_max_len([f[f"{prefix}_input_ids"] for f in features], self.pad_id)
            batch[f"{prefix}_attention_mask"] = pad_to_max_len([f[f"{prefix}_attention_mask"] for f in features], 0)
            batch[f"{prefix}_labels"] = pad_to_max_len([f[f"{prefix}_labels"] for f in features], -100)
        if "ref_chosen_logp" in features[0]:
            batch["ref_chosen_logp"] = torch.tensor([f["ref_chosen_logp"] for f in features], dtype=torch.float32)
            batch["ref_rejected_logp"] = torch.tensor([f["ref_rejected_logp"] for f in features], dtype=torch.float32)
        return batch
