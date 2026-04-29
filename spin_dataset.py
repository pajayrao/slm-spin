import logging
from typing import List, Dict
from torch.utils.data import Dataset
from spin_config import *
from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)

class SPINDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, cfg: SPINConfig, ref_logprobs=None):
        # Pre-tokenize all rows upfront so workers don't need the tokenizer or raw text
        self.chosen = []
        self.rejected = []
        for row in rows:
            self.chosen.append(tokenize_prompt_response(tokenizer, row["prompt"], row["response"], cfg))
            self.rejected.append(tokenize_prompt_response(tokenizer, row["prompt"], row["synthetic_response"], cfg))
        # ref_logprobs: list of dicts or None
        self.ref_logprobs = ref_logprobs

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        chosen = self.chosen[idx]
        rejected = self.rejected[idx]
        item = {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            # Used by Trainer's LengthGroupedSampler (group_by_length=True) to sort
            # batches by sequence length, minimising padding waste.
            "length": max(len(chosen["input_ids"]), len(rejected["input_ids"])),
        }
        if self.ref_logprobs is not None:
            item["ref_chosen_logp"] = self.ref_logprobs[idx]["ref_chosen_logp"]
            item["ref_rejected_logp"] = self.ref_logprobs[idx]["ref_rejected_logp"]
        return item
