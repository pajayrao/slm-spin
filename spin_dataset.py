import logging
from typing import List, Dict
from torch.utils.data import Dataset
from spin_config import *
from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, cfg: SPINConfig, ref_logprobs=None):
        logger.info(f"SPINDataset.__init__() — pre-tokenizing {len(rows)} rows "
                    f"(max_prompt={cfg.max_prompt_length}, max_length={cfg.max_length})...")
        self.chosen = []
        self.rejected = []
        chosen_lens = []
        rejected_lens = []

        for i, row in enumerate(rows):
            c = tokenize_prompt_response(
                tokenizer, row["prompt"], row["response"], cfg)
            r = tokenize_prompt_response(
                tokenizer, row["prompt"], row["synthetic_response"], cfg)
            self.chosen.append(c)
            self.rejected.append(r)
            chosen_lens.append(len(c["input_ids"]))
            rejected_lens.append(len(r["input_ids"]))

            if (i + 1) % 2000 == 0:
                logger.info(f"  Tokenised {i + 1}/{len(rows)} rows...")

        self.ref_logprobs = ref_logprobs

        avg_c = sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0
        avg_r = sum(rejected_lens) / len(rejected_lens) if rejected_lens else 0
        max_c = max(chosen_lens) if chosen_lens else 0
        max_r = max(rejected_lens) if rejected_lens else 0
        logger.info(f"SPINDataset ready: {len(self.chosen)} examples.")
        logger.info(
            f"  Chosen  seq lengths — avg={avg_c:.1f}, max={max_c} tokens.")
        logger.info(
            f"  Rejected seq lengths — avg={avg_r:.1f}, max={max_r} tokens.")
        logger.info(f"  ref_logprobs attached: {ref_logprobs is not None} "
                    f"({'required for SPIN loss' if ref_logprobs is not None else 'absent — logprobs must come from batch'}).")

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
            # Used by Trainer's LengthGroupedSampler to sort batches by length, minimising padding.
            "length": max(len(chosen["input_ids"]), len(rejected["input_ids"])),
        }
        if self.ref_logprobs is not None:
            item["ref_chosen_logp"] = self.ref_logprobs[idx]["ref_chosen_logp"]
            item["ref_rejected_logp"] = self.ref_logprobs[idx]["ref_rejected_logp"]
        return item
