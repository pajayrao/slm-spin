import logging
import os
import torch
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from spin_config import *
from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


class SPINDataset(Dataset):
    """PyTorch Dataset that pre-tokenizes chosen/rejected pairs for SPIN training.

    On first construction the rows are tokenized and (optionally) saved to a .pt
    cache file so subsequent runs with the same batch skip the tokenization step.

    Example (__init__):
        Input:  rows=[
                    {"prompt": "What is Python?", "response": "Python is a language.",
                     "synthetic_response": "Python is a scripting language..."},
                    {"prompt": "Explain AI.", "response": "AI stands for Artificial Intelligence.",
                     "synthetic_response": "AI is a computer science field..."},
                ]
                ref_logprobs=[
                    {"ref_chosen_logp": -12.43, "ref_rejected_logp": -18.07},
                    {"ref_chosen_logp":  -9.82, "ref_rejected_logp": -14.55},
                ]
                cache_path="output/synth_cache/iter_0_batch_000000_tokenized.pt"

        Output: SPINDataset with len=2; self.chosen and self.rejected lists of tokenized dicts;
                .pt cache written for faster restart; logs sequence length stats.
    """
    def __init__(self, rows: List[Dict[str, str]], tokenizer, cfg: SPINConfig,
                 ref_logprobs=None, cache_path: Optional[str] = None):
        if cache_path and os.path.exists(cache_path):
            logger.info(f"SPINDataset.__init__() — loading tokenized cache from {cache_path}...")
            cached = torch.load(cache_path, weights_only=False)
            self.chosen = cached["chosen"]
            self.rejected = cached["rejected"]
            chosen_lens = [len(c["input_ids"]) for c in self.chosen]
            rejected_lens = [len(r["input_ids"]) for r in self.rejected]
        else:
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

            if cache_path:
                tmp = cache_path + ".tmp"
                torch.save({"chosen": self.chosen, "rejected": self.rejected}, tmp)
                os.replace(tmp, cache_path)
                logger.info(f"  Tokenized data cached → {cache_path}")

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
        """Return the number of training examples in the dataset.

        Example:
            Input:  dataset constructed from 500 rows
            Output: 500
        """
        return len(self.chosen)

    def __getitem__(self, idx):
        """Return the tokenized dict for example at index idx.

        Example:
            Input:  idx=0  (first example — "What is Python?")

            Output: {
                "chosen_input_ids":       [1, 1724, 338, 5132, 29973, 5132, 338, ...],  # prompt+human response
                "chosen_attention_mask":  [1, 1, 1, 1, 1, 1, 1, ...],                  # all 1s (no padding)
                "chosen_labels":          [-100, -100, -100, -100, -100, 5132, 338, ...], # prompt masked
                "rejected_input_ids":     [1, 1724, 338, 5132, 29973, 5132, 338, ...],  # prompt+synthetic resp
                "rejected_attention_mask":[1, 1, 1, 1, 1, 1, 1, ...],
                "rejected_labels":        [-100, -100, -100, -100, -100, 5132, 338, ...],
                "length":                 148,  # max(len(chosen_input_ids), len(rejected_input_ids))
                "ref_chosen_logp":        -12.43,   # only present if ref_logprobs was provided
                "ref_rejected_logp":      -18.07,
            }
        """
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
