"""
sft_warmup.py — Optional supervised fine-tuning (SFT) warmup stage for SPIN.

Reproduces the precondition the SPIN paper relies on. In the paper SPIN starts
from `zephyr-7b-sft-full` — a base model already SFT'd on the *same* dataset
(UltraChat200k) that SPIN then draws its gold responses from. That "SFT'd-but-
underfit on p_data" seed is what lets self-play *sharpen* the model instead of
dragging it through a large distribution shift.

Off-the-shelf base/instruct checkpoints are not SFT'd on your gold data, so this
module lets you reproduce the recipe for **any** model:

    base model  →  run_sft_warmup() (1 epoch on prompt/response)  →  SPIN

The warmed-up model is saved to cfg.sft_warmup_dir and used as iteration 0's
starting model (π_0) instead of cfg.model_name_or_path.

Everything reuses the existing SPIN infrastructure so formatting stays identical
to what the SPIN trainer expects:
  - tokenize_prompt_response()  — same chat-template / truncation / label masking
  - load_causal_lm / make_trainable / merge_lora_and_get_base  — same model path
  - build_training_args         — same TrainingArguments construction
Only the loss differs: this stage uses the model's built-in next-token
cross-entropy (standard SFT), not the SPIN margin loss.
"""

import os
import shutil
import dataclasses
import logging
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from spin_config import SPINConfig
from utils import *

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """PyTorch Dataset that pre-tokenizes (prompt, response) rows for SFT warmup.

    Each row is tokenized with tokenize_prompt_response() — the exact same helper
    SPINDataset uses — so the chat template, left-truncation to cfg.max_length, and
    prompt-position masking (-100 in labels) match the SPIN pipeline byte-for-byte.

    Example (__init__):
        Input:  rows=[
                    {"prompt": "What is Python?", "response": "Python is a language."},
                    {"prompt": "Explain AI.",     "response": "AI is a CS field."},
                ]
                cache_path="output/synth_cache/sft_warmup_tokenized.pt"

        Output: SFTDataset with len=2; self.examples = list of
                {"input_ids": [...], "attention_mask": [...], "labels": [...]} dicts
                (labels have prompt positions set to -100); .pt cache written.
    """

    def __init__(self, rows: List[Dict[str, str]], tokenizer, cfg: SPINConfig,
                 cache_path: Optional[str] = None):
        # Step 1: Load the tokenized cache if present — skips re-tokenizing large
        # datasets (350k rows can take many minutes). Mirrors SPINDataset caching.
        if cache_path and os.path.exists(cache_path):
            logger.info(f"SFTDataset.__init__() — loading tokenized cache from {cache_path}...")
            self.examples = torch.load(cache_path, weights_only=False)["examples"]
        else:
            # Step 2: Tokenize every (prompt, response) pair into SFT tensors.
            logger.info(f"SFTDataset.__init__() — pre-tokenizing {len(rows)} rows "
                        f"(max_prompt={cfg.max_prompt_length}, max_length={cfg.max_length})...")
            self.examples = []
            for i, row in enumerate(rows):
                self.examples.append(
                    tokenize_prompt_response(tokenizer, row["prompt"], row["response"], cfg))
                if (i + 1) % 5000 == 0:
                    logger.info(f"  Tokenised {i + 1}/{len(rows)} rows...")

            # Step 3: Atomically write the tokenized cache (.tmp then os.replace).
            if cache_path:
                ensure_dir(os.path.dirname(cache_path))
                tmp = cache_path + ".tmp"
                torch.save({"examples": self.examples}, tmp)
                os.replace(tmp, cache_path)
                logger.info(f"  Tokenized SFT data cached → {cache_path}")

        # Step 4: Log sequence-length stats to diagnose truncation.
        lens = [len(e["input_ids"]) for e in self.examples]
        avg = sum(lens) / len(lens) if lens else 0
        n_resp = [sum(1 for t in e["labels"] if t != -100) for e in self.examples]
        avg_resp = sum(n_resp) / len(n_resp) if n_resp else 0
        logger.info(f"SFTDataset ready: {len(self.examples)} examples. "
                    f"seq len avg={avg:.1f}, max={max(lens) if lens else 0}; "
                    f"response tokens avg={avg_resp:.1f}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataCollator:
    """Right-pad a batch of SFT examples into stacked tensors.

    input_ids are padded with the tokenizer pad id, attention_mask with 0, and
    labels with -100 (so padded positions are ignored by the cross-entropy loss).

    Example:
        Input:  features=[
                    {"input_ids":[1,5,8], "attention_mask":[1,1,1], "labels":[-100,5,8]},
                    {"input_ids":[1,9],   "attention_mask":[1,1],   "labels":[-100,9]},
                ]
                tokenizer.pad_token_id=0

        Output: {
            "input_ids":      tensor([[1,5,8],[1,9,0]]),
            "attention_mask": tensor([[1,1,1],[1,1,0]]),
            "labels":         tensor([[-100,5,8],[-100,9,-100]]),
        }
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":      pad_to_max_len([f["input_ids"] for f in features], self.pad_id),
            "attention_mask": pad_to_max_len([f["attention_mask"] for f in features], 0),
            "labels":         pad_to_max_len([f["labels"] for f in features], -100),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Warmup runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sft_warmup(cfg: SPINConfig, tokenizer, base_rows: List[Dict[str, str]],
                   base_model_path: Optional[str] = None,
                   out_dir: Optional[str] = None) -> str:
    """Run one SFT pass over base_rows and return the warmed-up model directory.

    base_model_path defaults to cfg.model_name_or_path and out_dir to
    cfg.sft_warmup_dir. Pass explicit values to SFT an arbitrary checkpoint into a
    dedicated directory — this is how the interleaved SFT stage re-fits each even
    SPIN iteration's incoming model (the previous iteration's checkpoint) to the
    gold data before that iteration's self-play.

    Idempotent: if a completed warmup checkpoint already exists (a `.done` sentinel
    plus config.json in out_dir), this returns immediately so resuming a SPIN run
    never re-trains a finished SFT pass. Otherwise it loads base_model_path,
    SFTs it (LoRA or full, per cfg.use_lora), merges any LoRA adapters, and saves a
    plain AutoModelForCausalLM that load_causal_lm() can load like any checkpoint.

    Example (fresh run):
        Input:  cfg.model_name_or_path="Qwen/Qwen2.5-0.5B",
                cfg.sft_warmup_dir="./spin_outputs/sft_warmup",
                cfg.sft_warmup_epochs=1.0, cfg.sft_warmup_learning_rate=2e-5,
                base_rows=<50000 {prompt, response} dicts>

        Output: "./spin_outputs/sft_warmup"  (merged model + tokenizer + .done written)

    Example (already complete — resume):
        Input:  cfg.sft_warmup_dir contains config.json and .done
        Output: "./spin_outputs/sft_warmup"  (returns immediately, no training)
    """
    base_model_path = base_model_path or cfg.model_name_or_path
    out_dir = out_dir or cfg.sft_warmup_dir
    ensure_dir(out_dir)
    done_path = os.path.join(out_dir, ".done")

    if os.path.exists(done_path) and os.path.exists(os.path.join(out_dir, "config.json")):
        logger.info(f"run_sft_warmup() — warmup already complete at {out_dir}; skipping.")
        return out_dir

    # Step 1: Select the warmup rows (optionally capped; base_rows is already
    # bounded by cfg.max_data_load).
    rows = base_rows
    if cfg.sft_warmup_max_samples and cfg.sft_warmup_max_samples > 0:
        rows = base_rows[:cfg.sft_warmup_max_samples]
    logger.info(
        f"run_sft_warmup() — SFT warmup starting: {len(rows)} rows, "
        f"epochs={cfg.sft_warmup_epochs}, lr={cfg.sft_warmup_learning_rate:.2e}, "
        f"use_lora={cfg.use_lora}, base={base_model_path}")

    # Step 2: Build the tokenized dataset + collator (same formatting as SPIN).
    # The cache filename embeds the tokenization fingerprint: this cache was
    # previously keyed by a fixed name, so switching model family (different
    # tokenizer!) or sequence settings silently reused stale tensors. The tokenized
    # data depends only on tokenizer + formatting settings (all captured by the
    # fingerprint), not on which checkpoint is being warmed, so repeated SFT passes
    # within one run still share a single cache.
    cache_path = None
    if cfg.sft_warmup_cache_tokenized:
        cache_path = os.path.join(
            cfg.synthetic_cache_dir,
            f"sft_warmup_tokenized_{tokenization_fingerprint(cfg)}.pt")
    dataset = SFTDataset(rows, tokenizer, cfg, cache_path=cache_path)
    collator = SFTDataCollator(tokenizer)

    # Step 3: Load the base model and make it trainable (LoRA or full fine-tune).
    log_memory("before_load_sft_warmup")
    model = load_causal_lm(base_model_path, cfg, trainable=False).to(cfg.device)
    model = make_trainable(model, cfg)
    log_memory("after_load_sft_warmup")

    # Step 4: TrainingArguments — reuse build_training_args, overriding only the
    # epoch count and the checkpoint policy. save_strategy="steps" (save_total_limit=1)
    # lets a long warmup resume after a thermal shutdown, consistent with the rest of
    # the SPIN pipeline. The warmup LR is passed explicitly so it stays independent of
    # the SPIN iteration LRs.
    tb_log_dir = os.path.join(
        cfg.tensorboard_dir, os.path.basename(os.path.normpath(out_dir)))
    args_cfg = dataclasses.replace(
        cfg,
        num_epochs_per_iteration=cfg.sft_warmup_epochs,
        save_strategy="steps",
        save_total_limit=1,
    )
    args = build_training_args(
        args_cfg, out_dir, cfg.sft_warmup_learning_rate, logging_dir=tb_log_dir)

    # Step 5: Standard HF Trainer — labels are in the batch, so the model's built-in
    # causal-LM cross-entropy is the SFT loss (no custom trainer needed).
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Resume from a mid-training HF checkpoint if the warmup was interrupted.
    # Mirrors _step_train: a torch.compile wrapper makes HF treat the model as a full
    # model, so a PEFT-only checkpoint (adapter weights, no full-model index) can't be
    # loaded — in that case delete it and restart the warmup from scratch.
    resume_ckpt = get_last_checkpoint(out_dir)
    if resume_ckpt:
        _FULL_MODEL_FILES = [
            "model.safetensors", "pytorch_model.bin",
            "model.safetensors.index.json", "pytorch_model.bin.index.json",
        ]
        has_full_model = any(
            os.path.exists(os.path.join(resume_ckpt, f)) for f in _FULL_MODEL_FILES)
        if not has_full_model:
            logger.warning(
                f"  Warmup checkpoint {resume_ckpt} has only PEFT adapter weights "
                f"(no full-model index); cannot resume under torch.compile — "
                f"removing it and restarting the warmup from scratch.")
            shutil.rmtree(resume_ckpt)
            resume_ckpt = None
        else:
            logger.info(f"  Resuming SFT warmup from checkpoint: {resume_ckpt}")

    log_memory("before_sft_warmup_train")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    log_memory("after_sft_warmup_train")

    # Step 6: Merge LoRA into the base weights (no-op if full fine-tune), save a plain
    # model + tokenizer, remove the HF checkpoint-N dirs (the merged model supersedes
    # them), then write the .done sentinel.
    model = merge_lora_and_get_base(model, cfg)
    trainer.model = model
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    for entry in os.scandir(out_dir):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            shutil.rmtree(entry.path)
            logger.info(f"  Removed warmup HF checkpoint: {entry.path}")
    open(done_path, "w").close()
    logger.info(f"run_sft_warmup() — warmup model saved → {out_dir}, .done written.")

    free_model(model)
    return out_dir
