"""
evaluate.py — Post-training benchmark evaluation and TensorBoard logging for SPIN.

Evaluates six standard LLM benchmarks directly using HuggingFace transformers —
no lm_eval dependency. Multiple-choice tasks (ARC, TruthfulQA-MC2, Winogrande,
HellaSwag, MMLU) use log-likelihood continuation scoring; GSM8k uses greedy
generation with answer extraction.

Flow overview
-------------
  0. run_eval() is invoked once per SPIN iteration by main.py (right after that
     iteration's checkpoint is saved) when cfg.eval_run_after_training is True,
     as well as standalone via this file's CLI.
  1. Discover iteration checkpoint directories (iter_0, iter_1, …) under --checkpoints-dir.
  2. For each iteration:
       a. Skip straight to the cached .parsed.json if one already exists —
          this is what makes repeated per-iteration calls cheap.
       b. Otherwise load model + tokenizer from the checkpoint.
       c. Run all six benchmarks, collecting one scalar metric per task.
       d. Compute delta vs the previous iteration's scores.
       e. Track the running best-average across all evaluated iterations.
       f. Save per-iteration results to a JSON file.
       g. Free the model from GPU memory before loading the next checkpoint.
  3. Write a human-readable comparative summary (TXT + JSON).
  4. Write TensorBoard scalars, text cards, and metadata for the full run.

Usage
-----
  python evaluate.py
  python evaluate.py --iters iter_0 iter_2
  python evaluate.py --limit 50                          # smoke test
  python evaluate.py --tasks arc_challenge gsm8k         # subset of tasks
  python evaluate.py --skip-tasks hellaswag mmlu         # exclude slow tasks
  python evaluate.py --n-shots arc_challenge=10 gsm8k=3  # override shot counts
"""

import argparse
import logging
import os
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Optional
from tqdm import tqdm


import torch
import torch.nn.functional as F
from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.tensorboard import SummaryWriter

import dataclasses
from spin_config import SPINConfig
from utils import *

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_CHECKPOINT_SUBDIR_CANDIDATES: list[str] = [
    "hf_final", "final_checkpoint", "checkpoint-final", "merged", "model", ".",
]
_MODEL_CONFIG_MARKERS: list[str] = ["config.json", "adapter_config.json"]
_TRUST_REMOTE_CODE: bool = False
_TRUST_REMOTE_CODE_DATASETS: bool = False

# ---------------------------------------------------------------------------
# Generation (GSM8k)
# ---------------------------------------------------------------------------

GSM8K_MAX_NEW_TOKENS: int = 256

# ---------------------------------------------------------------------------
# TensorBoard tag strings
# ---------------------------------------------------------------------------

_TB_TAG_AVG              = "eval/average"
_TB_TAG_BEST_AVG         = "eval/best_so_far_average"
_TB_TAG_TASK_COUNT       = "eval/task_count"
_TB_TAG_TASKS            = "eval/tasks"
_TB_TAG_SCORECARD        = "eval/scorecard"
_TB_TAG_BEST_ITER        = "eval/best_iteration"
_TB_TAG_RUN_CONFIG       = "eval/run_config"
_TB_TAG_DELTA_PREFIX     = "compare_vs_prev/"
_TB_TAG_IMPROVED_COUNT   = "compare_vs_prev/improved_task_count"
_TB_TAG_DECLINED_COUNT   = "compare_vs_prev/declined_task_count"
_TB_TAG_IMPROVEMENT_RATE = "compare_vs_prev/improvement_rate"

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

_ITER_NUM_SENTINEL: int = 10**9

# Maximum total token length (context + continuation) fed to the model.
MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# Task registry — single source of truth
# ---------------------------------------------------------------------------
# Each row: (task_id, display_label, metric_type, default_shots, dataset_limit)
#
#   task_id       — key into _EVAL_FNS; accepted by --tasks / --skip-tasks
#   display_label — human-readable name; also accepted by --tasks / --skip-tasks
#   metric_type   — "acc" | "acc_norm" | "mc2"
#   default_shots — Open LLM Leaderboard v1 standard; change makes scores
#                   incomparable to published results
#   dataset_limit — max rows to evaluate (None = full set / global --limit)
TASKS: list[tuple[str, str, str, int, Optional[int]]] = [
    ("arc_challenge",  "Arc",        "acc_norm", 25, None),
    ("truthfulqa_mc2", "TruthfulQA", "mc2",       0, None),
    ("winogrande",     "Winogrande", "acc",        5, None),
    # ("gsm8k",          "GSM8k",      "acc",        5, None),
    ("hellaswag",      "HellaSwag",  "acc_norm",  10, None),
    ("mmlu",           "MMLU",       "acc",        5, None),
]

# Derived lookups — edit the TASKS list above; these stay in sync automatically.
DEFAULT_SHOTS:  dict[str, int]           = {tid: shots for tid, _, _, shots, _   in TASKS}
DATASET_LIMITS: dict[str, Optional[int]] = {tid: lim   for tid, _, _, _,     lim in TASKS}

# Quick lookup: is a given key a task label (vs "Average")?
_TASK_LABELS: set[str] = {label for _, label, *_ in TASKS}


def resolve_task_filter(
    names: list[str],
    tasks: list[tuple] = TASKS,
) -> list[tuple]:
    """Return the subset of *tasks* matched by *names* (IDs or labels, case-insensitive).

    Accepted forms (all equivalent for ARC):
        arc_challenge   ARC_CHALLENGE   Arc   arc

    Raises SystemExit listing every unrecognised name alongside the full valid set.
    """
    lookup: dict[str, tuple] = {}
    for task in tasks:
        task_id, label = task[0], task[1]
        lookup[task_id.lower()] = task
        lookup[label.lower()]   = task

    resolved: list[tuple] = []
    unknown:  list[str]   = []
    seen:     set[str]    = set()
    for name in names:
        key = name.lower()
        if key in lookup:
            task = lookup[key]
            if task[0] not in seen:
                resolved.append(task)
                seen.add(task[0])
        else:
            unknown.append(name)

    if unknown:
        pairs = ", ".join(f"{t[0]} ({t[1]})" for t in tasks)
        raise SystemExit(
            f"Unknown task(s): {unknown}\n"
            f"Valid tasks (id / label): {pairs}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Standard GSM8k CoT few-shot exemplars
# ---------------------------------------------------------------------------

_GSM8K_FEW_SHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. #### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates and her sister 42. Total = 32 + 42 = 74. "
        "After eating 35: 74 - 35 = 39. #### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. He now has 12. So he gave 20 - 12 = 8. #### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. He got 2 from mom and 2 from dad = 4 more. 5 + 4 = 9. #### 9",
    ),
]


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def find_model_path(iter_dir: Path) -> Path:
    """Locate the HuggingFace model directory inside an iteration checkpoint folder."""
    logger.info(f"Searching for model root in: {iter_dir}")
    for cand in _CHECKPOINT_SUBDIR_CANDIDATES:
        p = iter_dir / cand
        if not p.is_dir():
            continue
        if any((p / m).exists() for m in _MODEL_CONFIG_MARKERS):
            logger.info(f"  Found model root at candidate '{cand}': {p.resolve()}")
            return p.resolve()
        if cand == "." and any((iter_dir / m).exists() for m in _MODEL_CONFIG_MARKERS):
            logger.info(f"  Found model root at iteration directory itself: {iter_dir.resolve()}")
            return iter_dir.resolve()

    logger.info("  No candidate matched — falling back to recursive search.")
    for marker in _MODEL_CONFIG_MARKERS:
        found = list(iter_dir.rglob(marker))
        if found:
            logger.info(f"  Recursive search found {marker} at: {found[0].parent.resolve()}")
            return found[0].parent.resolve()

    logger.warning(f"  Could not locate a model config under {iter_dir}; will try iter_dir directly.")
    return iter_dir.resolve()


# ---------------------------------------------------------------------------
# Core log-likelihood scoring primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_continuations_batched(
    model,
    tokenizer,
    context: str,
    continuations: list[str],
    device: str,
) -> list[float]:
    """Score all continuations for one context in a single batched forward pass.

    Sequences are left-padded to the batch maximum length so that real tokens
    are right-aligned and attention patterns are valid across the batch.
    """
    if not continuations:
        return []

    pad_id = tokenizer.pad_token_id
    seq_data: list[tuple[list[int], int, int]] = []

    # Step 1: Tokenize each context+continuation pair and locate where the continuation starts.
    # Tokenize the full string (not separately) to avoid subword boundary artifacts.
    # cont_start = index of first continuation token inside full_ids.
    # Example: context="The cat", cont=" sat" → full_ids=[1,450,6635,3290], cont_len=2
    #          cont_start = 4 - 2 = 2  (tokens at positions 2,3 are the continuation)
    # If the full sequence exceeds MAX_SEQ_LEN, left-truncate and recalculate cont_start.
    for cont in continuations:
        full_ids: list[int] = tokenizer(context + cont, add_special_tokens=True)["input_ids"]
        cont_raw: list[int] = tokenizer(cont, add_special_tokens=False)["input_ids"]
        cont_len = len(cont_raw)
        if cont_len == 0:
            seq_data.append(([], 0, 0))
            continue
        cont_start = len(full_ids) - cont_len
        if len(full_ids) > MAX_SEQ_LEN:
            full_ids = full_ids[-MAX_SEQ_LEN:]
            cont_start = max(1, len(full_ids) - cont_len)
        seq_data.append((full_ids, cont_start, cont_len))

    # Step 2: Left-pad all sequences to the same length for batched GPU processing.
    # Real tokens are right-aligned; padding goes on the left with attention_mask=0.
    # adj_start shifts cont_start right by pad_len to stay aligned after padding.
    # Example: max_len=5, seq A has 3 tokens → pad_len=2
    #          padded_ids[A] = [pad, pad, tok0, tok1, tok2]
    #          attn_mask[A]  = [0,   0,   1,    1,    1  ]
    #          adj_start[A]  = cont_start + 2
    max_len = max((len(d[0]) for d in seq_data), default=0)
    padded_ids, attn_masks, adj_starts = [], [], []

    for full_ids, cont_start, _ in seq_data:
        if not full_ids:
            padded_ids.append([pad_id] * max_len)
            attn_masks.append([0] * max_len)
            adj_starts.append(0)
            continue
        pad_len = max_len - len(full_ids)
        padded_ids.append([pad_id] * pad_len + full_ids)
        attn_masks.append([0] * pad_len + [1] * len(full_ids))
        adj_starts.append(cont_start + pad_len)

    # Step 3: Single batched forward pass → log-probabilities over vocab at every position.
    # log_softmax converts raw logits to normalized log-probs along the vocab dimension.
    # Shape: (num_continuations, max_len, vocab_size)
    input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attn_mask_t = torch.tensor(attn_masks,  dtype=torch.long, device=device)
    logits      = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits
    # log_softmax is applied per-row on just the continuation slice in Step 4 below, NOT over
    # the full (rows, max_len, vocab) tensor — the per-position vocab projection dominates
    # eval memory with large vocabularies (e.g. Qwen ~152k), so the full-sequence softmax is
    # the main eval OOM source on small GPUs.

    # Step 4: For each continuation, extract the log-prob of each actual token and sum.
    # Autoregressive alignment: the prediction for position cs is at log_probs[cs-1],
    # because the model at step t predicts step t+1.
    # Example: cont_tok_ids=[3290, 1234], adj_start=2
    #          pred_lp = log_probs[i, 1:3]       ← positions cs-1 and cs
    #          score   = pred_lp[0,3290] + pred_lp[1,1234]  ← log p(token) at each step
    scores: list[float] = []
    for i, (full_ids, _, cont_len) in enumerate(seq_data):
        if not full_ids or cont_len == 0:
            scores.append(0.0)
            continue
        cs = adj_starts[i]
        actual_len = min(cont_len, max_len - cs)
        cont_tok_ids = input_ids_t[i, cs:cs + actual_len]
        pred_lp      = F.log_softmax(logits[i, cs - 1:cs - 1 + actual_len], dim=-1)
        scores.append(pred_lp[torch.arange(actual_len, device=device), cont_tok_ids].sum().item())

    return scores


@torch.no_grad()
def score_examples_batched(
    model,
    tokenizer,
    examples: list[tuple[str, list[str]]],
    device: str,
    rows_per_batch: int = 32,
) -> list[list[float]]:
    """Score many (context, continuations) pairs with cross-example GPU batching.

    More efficient than calling score_continuations_batched() per example because
    it amortizes kernel-launch overhead and achieves higher GPU utilisation.
    Each GPU forward pass handles rows_per_batch (context+continuation) sequences
    regardless of how many examples or choices they span.
    """
    pad_id = tokenizer.pad_token_id
    output: list[list[float]] = [[0.0] * len(conts) for _, conts in examples]

    # Step 1: Flatten all (context, continuation) pairs from every example into one list.
    # Each entry records full_ids, cont_start, cont_len plus (ex_idx, cont_idx) so scores
    # can be written back to the right slot in output after the forward pass.
    # Example: examples=[("The cat", [" sat", " ran"]), ("A dog", [" barked"])]
    #          flat = [(full_ids_0, cs_0, cl_0, 0, 0),   ← "The cat sat"
    #                  (full_ids_1, cs_1, cl_1, 0, 1),   ← "The cat ran"
    #                  (full_ids_2, cs_2, cl_2, 1, 0)]   ← "A dog barked"
    flat: list[tuple[list[int], int, int, int, int]] = []
    for ex_idx, (context, continuations) in enumerate(examples):
        for cont_idx, cont in enumerate(continuations):
            full_ids: list[int] = tokenizer(context + cont, add_special_tokens=True)["input_ids"]
            cont_len = len(tokenizer(cont, add_special_tokens=False)["input_ids"])
            if cont_len == 0:
                continue
            cont_start = len(full_ids) - cont_len
            if len(full_ids) > MAX_SEQ_LEN:
                full_ids = full_ids[-MAX_SEQ_LEN:]
                cont_start = max(1, len(full_ids) - cont_len)
            flat.append((full_ids, cont_start, cont_len, ex_idx, cont_idx))

    # Step 2: Process flat list in GPU-sized chunks (rows_per_batch rows per forward pass).
    # Within each chunk, left-pad to chunk-max length, run the model once, then gather scores.
    # Left-padding keeps real tokens right-aligned so attention is computed correctly.
    # Example: chunk has 3 rows with lengths [6, 4, 5], max_len=6
    #          row 1 (len 4): padded=[pad,pad,t0,t1,t2,t3], mask=[0,0,1,1,1,1], adj_cs += 2
    for i in range(0, len(flat), rows_per_batch):
        batch = flat[i : i + rows_per_batch]
        max_len = max(len(r[0]) for r in batch)

        padded, masks, adj_cs = [], [], []
        for full_ids, cont_start, *_ in batch:
            pl = max_len - len(full_ids)
            padded.append([pad_id] * pl + full_ids)
            masks.append([0] * pl + [1] * len(full_ids))
            adj_cs.append(cont_start + pl)

        # Step 3: Single forward pass for the whole chunk → log-probs over vocab.
        ids_t  = torch.tensor(padded, dtype=torch.long, device=device)
        mask_t = torch.tensor(masks,  dtype=torch.long, device=device)
        logits = model(input_ids=ids_t, attention_mask=mask_t).logits

        # Step 4: Gather per-token log-probs for the continuation slice and sum into a scalar.
        # Autoregressive offset: prediction for position cs is stored at lp[cs-1].
        # Write score back to output[ex_idx][cont_idx] so the outer loop gets a 2-D result.
        # Example: cs=2, cont=[tok_a, tok_b]
        #          score = lp[j, 1, tok_a] + lp[j, 2, tok_b]
        # log_softmax runs per-row over ONLY the continuation positions (cs-1 : cs-1+alen),
        # never the full (rows, max_len, vocab) tensor — that full-vocab softmax is the
        # dominant eval memory cost on large vocabularies (Qwen ~152k) and small GPUs.
        for j, (full_ids, _, cont_len, ex_idx, cont_idx) in enumerate(batch):
            cs = adj_cs[j]
            alen = min(cont_len, max_len - cs)
            if alen <= 0:
                continue
            tok = ids_t[j, cs : cs + alen]
            row_lp = F.log_softmax(logits[j, cs - 1 : cs - 1 + alen], dim=-1)
            score = row_lp[torch.arange(alen, device=device), tok].sum().item()
            output[ex_idx][cont_idx] = score

    return output


def _pick_best(scores: list[float], choices: list[str], normalize: bool) -> int:
    """Select the index of the highest-scoring choice.

    When normalize=True (acc_norm) each score is divided by character length of
    the choice text to prevent bias toward shorter answers.
    """
    # Step 1: Optionally normalize each score by the character length of the choice text.
    # This prevents the model from preferring shorter answers just because fewer tokens
    # means fewer log-probs to sum (shorter sums are less negative).
    # Example: scores=[-3.0, -2.1], choices=["Paris", "A large city in France"]
    #          raw        → pick index 1 (-2.1 > -3.0) ← wrong, length bias
    #          normalized → [-3.0/5, -2.1/22] = [-0.60, -0.095] → pick index 1 still,
    #                       but score is now per-character so long/short compete fairly
    if normalize:
        adjusted = [s / max(len(c), 1) for s, c in zip(scores, choices)]
    else:
        adjusted = list(scores)
    # Step 2: Return the index of the highest (least-negative) adjusted score.
    return int(max(range(len(adjusted)), key=lambda i: adjusted[i]))


# ---------------------------------------------------------------------------
# ARC Challenge  (metric: acc_norm)
# ---------------------------------------------------------------------------

def eval_arc_challenge(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate ARC-Challenge using length-normalised log-likelihood (acc_norm)."""
    logger.info(f"ARC-Challenge: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  ARC-Challenge: {len(test_examples)} test examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            labels = ex["choices"]["label"]
            texts  = ex["choices"]["text"]
            answer_text = texts[labels.index(ex["answerKey"])]
            few_shot_prefix += f"Question: {ex['question']}\nAnswer: {answer_text}\n\n"

    logger.info("  ARC-Challenge: scoring examples...")
    inputs = [
        (few_shot_prefix + f"Question: {ex['question']}\nAnswer:", [f" {t}" for t in ex["choices"]["text"]])
        for ex in test_examples
    ]
    all_scores = score_examples_batched(model, tokenizer, inputs, device, batch_size)

    correct = 0
    for ex_scores, ex, (_, choices) in zip(all_scores, test_examples, inputs):
        labels = ex["choices"]["label"]
        if labels[_pick_best(ex_scores, choices, normalize=True)] == ex["answerKey"]:
            correct += 1
    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# TruthfulQA MC2  (metric: mc2)
# ---------------------------------------------------------------------------

def eval_truthfulqa_mc2(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate TruthfulQA MC2: softmax probability mass on correct answers."""
    logger.info(f"TruthfulQA MC2: loading dataset (zero-shot, limit={limit})")
    if n_shot != 0:
        logger.warning(f"TruthfulQA has no few-shot train split; n_shot={n_shot} ignored.")
    ds = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    examples = list(ds["validation"])
    if limit:
        examples = examples[:limit]
    logger.info(f"  TruthfulQA MC2: {len(examples)} validation examples.")

    inputs = [
        (f"Q: {ex['question']}\nA:", [f" {c}" for c in ex["mc2_targets"]["choices"]])
        for ex in examples
    ]
    all_scores = score_examples_batched(model, tokenizer, inputs, device, batch_size)

    mc2_scores: list[float] = []
    for ex_scores, ex in zip(all_scores, examples):
        labels  = ex["mc2_targets"]["labels"]

        # Step 1: Convert raw log-likelihood scores to a proper probability distribution.
        # softmax re-normalizes across ALL answer choices (both correct and incorrect),
        # so the values sum to 1 and are comparable across questions with different
        # numbers of choices.
        # Example: ex_scores=[-2.1, -3.4, -1.8, -4.0]  (4 choices)
        #          probs ≈ [0.31, 0.09, 0.42, 0.04, ...] after softmax (sums to 1.0)
        log_lls = torch.tensor(ex_scores, dtype=torch.float64)
        probs   = torch.softmax(log_lls, dim=0)

        # Step 2: MC2 score = sum of probabilities assigned to the *correct* answers.
        # labels[i]==1 marks a correct answer; labels[i]==0 marks an incorrect one.
        # Example: labels=[0,1,1,0], probs=[0.31,0.09,0.42,0.04]
        #          mc2 = probs[1] + probs[2] = 0.09 + 0.42 = 0.51
        #          (higher = model assigns more probability mass to truthful answers)
        mc2_scores.append(float(sum(probs[i] for i, lbl in enumerate(labels) if lbl == 1)))
    return float(sum(mc2_scores) / len(mc2_scores))


# ---------------------------------------------------------------------------
# Winogrande  (metric: acc)
# ---------------------------------------------------------------------------

def eval_winogrande(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate Winogrande commonsense pronoun resolution (acc)."""
    logger.info(f"Winogrande: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("winogrande", "winogrande_xl", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["validation"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  Winogrande: {len(test_examples)} validation examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            answer_option = ex["option1"] if ex["answer"] == "1" else ex["option2"]
            few_shot_prefix += ex["sentence"].replace("_", answer_option) + "\n\n"

    # Step 1: Split each sentence at the blank ("_") into context and remainder.
    # Winogrande sentences have exactly one "_" placeholder for a pronoun or noun.
    # The two candidate options (option1, option2) each fill that blank.
    # We reconstruct full sentences by inserting each option and appending the rest
    # of the original sentence after the blank so the model scores the complete text.
    # Example: sentence="The trophy wouldn't fit in the brown suitcase because _ was too big."
    #          blank_idx = 46  (index of "_")
    #          context = "The trophy wouldn't fit in the brown suitcase because "
    #          rest    = " was too big."
    #          option1="it", option2="the suitcase"
    #          continuations = ["it was too big.", "the suitcase was too big."]
    inputs = []
    for ex in test_examples:
        blank_idx = ex["sentence"].index("_")
        context   = few_shot_prefix + ex["sentence"][:blank_idx]
        rest      = ex["sentence"][blank_idx + 1:]
        inputs.append((context, [ex["option1"] + rest, ex["option2"] + rest]))

    all_scores = score_examples_batched(model, tokenizer, inputs, device, batch_size)

    # Step 2: Pick the higher-scoring option and compare against the ground-truth answer label.
    # answer is "1" or "2" (string), so we map score comparison to the same string format.
    # No length normalization here (unlike ARC/HellaSwag) because both options fill the same
    # blank in the same sentence frame, so their lengths are directly comparable.
    # Example: s1=-2.1 (score for option1 "it"), s2=-3.4 (score for option2 "the suitcase")
    #          s1 > s2 → predicted "1"; answer="1" → correct
    correct = sum(
        1 for (s1, s2), ex in zip(all_scores, test_examples)
        if ("1" if s1 > s2 else "2") == ex["answer"]
    )
    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# GSM8k  (metric: acc — exact numeric match)
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from generated or reference text."""
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def eval_gsm8k(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate GSM8k grade-school math via batched greedy generation (acc)."""
    logger.info(f"GSM8k: loading dataset (n_shot={n_shot}, limit={limit}, max_new_tokens={GSM8K_MAX_NEW_TOKENS})")
    ds = load_dataset("gsm8k", "main", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  GSM8k: {len(test_examples)} test examples.")

    # Step 1: Build a fixed few-shot prefix from hand-curated Q&A pairs.
    # These examples are prepended to every test prompt to prime the model's output
    # format, particularly the "####<answer>" pattern that _extract_number looks for.
    # Example: n_shot=4 → prefix = "Question: ...\nAnswer: ...\n\nQuestion: ...\n..." (4 blocks)
    few_shot_prefix = "".join(
        f"Question: {q}\nAnswer: {a}\n\n"
        for q, a in _GSM8K_FEW_SHOT[:n_shot]
    )

    # Step 2: Reserve prompt token budget so the generation never exceeds MAX_SEQ_LEN.
    # max_prompt_len caps the tokenized prompt; the model then generates up to
    # GSM8K_MAX_NEW_TOKENS additional tokens for the answer. Together they stay
    # within the model's context window.
    # Example: MAX_SEQ_LEN=2048, GSM8K_MAX_NEW_TOKENS=256 → max_prompt_len=1792
    max_prompt_len = MAX_SEQ_LEN - GSM8K_MAX_NEW_TOKENS
    correct = 0
    for i in tqdm(range(0, len(test_examples), batch_size), desc="GSM8k", leave=False, unit="batch"):
        batch   = test_examples[i : i + batch_size]

        # Step 3: Tokenize prompts with right-padding so batched generation works correctly.
        # padding=True pads shorter prompts in the batch to the longest one.
        # truncation=True ensures no prompt exceeds max_prompt_len (drops from the left
        # implicitly via tokenizer default, keeping the question tail).
        # Moving to device here because model.generate() requires on-device tensors.
        # Example: batch of 8 questions → enc["input_ids"] shape (8, max_prompt_len)
        prompts = [few_shot_prefix + f"Question: {ex['question']}\nAnswer:" for ex in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Step 4: Greedy decoding (do_sample=False) to produce deterministic answers.
        # out shape: (batch, prompt_len + num_generated_tokens) — includes the prompt prefix.
        # prompt_len is constant across the batch after padding, so we slice [prompt_len:]
        # to isolate only the newly generated tokens.
        # Example: enc shape (8, 1792), max_new_tokens=256 → out shape (8, 2048)
        #          out[j, 1792:] contains the generated answer for example j
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Step 5: Decode generated tokens and compare extracted numbers to ground truth.
        # _extract_number first looks for the "#### <answer>" chain-of-thought marker,
        # then falls back to the last number in the string if the marker is absent.
        # Both generated text and the reference answer go through _extract_number so the
        # comparison is format-agnostic (commas, leading zeros, etc. are normalized away).
        # Example: generated="...so the answer is #### 42", answer="#### 42"
        #          _extract_number(generated)="42", _extract_number(answer)="42" → correct
        prompt_len = enc["input_ids"].shape[1]
        for j, ex in enumerate(batch):
            generated = tokenizer.decode(out[j, prompt_len:], skip_special_tokens=True)
            if _extract_number(generated) == _extract_number(ex["answer"]):
                correct += 1

    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# HellaSwag  (metric: acc_norm)
# ---------------------------------------------------------------------------

def _clean_hellaswag(text: str) -> str:
    """Strip bracketed annotation artifacts and collapse whitespace."""
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def eval_hellaswag(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate HellaSwag commonsense sentence completion (acc_norm)."""
    logger.info(f"HellaSwag: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("Rowan/hellaswag", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    val_examples = list(ds["validation"])
    if limit:
        val_examples = val_examples[:limit]
    logger.info(f"  HellaSwag: {len(val_examples)} validation examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            ctx         = _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"])
            best_ending = _clean_hellaswag(ex["endings"][int(ex["label"])])
            few_shot_prefix += f"{ctx} {best_ending}\n\n"

    inputs = [
        (
            few_shot_prefix + _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"]),
            [" " + _clean_hellaswag(e) for e in ex["endings"]],
        )
        for ex in val_examples
    ]
    all_scores = score_examples_batched(model, tokenizer, inputs, device, batch_size)

    correct = sum(
        1 for ex_scores, ex, (_, endings) in zip(all_scores, val_examples, inputs)
        if _pick_best(ex_scores, endings, normalize=True) == int(ex["label"])
    )
    return correct / len(val_examples)


# ---------------------------------------------------------------------------
# MMLU  (metric: acc)
# ---------------------------------------------------------------------------

_MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def _mmlu_format(ex: dict, with_answer: bool = False) -> str:
    """Render an MMLU example into the standard prompt format."""
    subj = ex.get("subject", "").replace("_", " ")
    header = f"The following is a multiple choice question about {subj}.\n" if subj else ""
    choices_str = "\n".join(
        f"{lbl}. {ex['choices'][i]}" for i, lbl in enumerate(_MMLU_CHOICE_LABELS)
    )
    text = header + f"{ex['question'].strip()}\n{choices_str}\nAnswer:"
    if with_answer:
        text += f" {_MMLU_CHOICE_LABELS[ex['answer']]}\n\n"
    return text


def eval_mmlu(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 4
) -> float:
    """Evaluate MMLU across 57 subjects using single-letter continuation scoring (acc)."""
    logger.info(f"MMLU: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("cais/mmlu", "all", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  MMLU: {len(test_examples)} test examples across all subjects.")

    dev_by_subject: dict[str, list] = {}
    for ex in ds.get("dev", []):
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    letter_choices = [f" {lbl}" for lbl in _MMLU_CHOICE_LABELS]

    inputs = []
    for ex in test_examples:
        few_shot_prefix = ""
        if n_shot > 0:
            for shot in dev_by_subject.get(ex.get("subject", ""), [])[:n_shot]:
                few_shot_prefix += _mmlu_format(shot, with_answer=True)
        inputs.append((few_shot_prefix + _mmlu_format(ex), letter_choices))

    all_scores = score_examples_batched(model, tokenizer, inputs, device, batch_size)

    correct = sum(
        1 for ex_scores, ex in zip(all_scores, test_examples)
        if _pick_best(ex_scores, letter_choices, normalize=False) == ex["answer"]
    )
    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# Benchmark dispatcher
# ---------------------------------------------------------------------------

_EVAL_FNS = {
    "arc_challenge":  eval_arc_challenge,
    "truthfulqa_mc2": eval_truthfulqa_mc2,
    "winogrande":     eval_winogrande,
    "gsm8k":          eval_gsm8k,
    "hellaswag":      eval_hellaswag,
    "mmlu":           eval_mmlu,
}


def run_all_benchmarks(
    model,
    tokenizer,
    device: str,
    n_shots: dict[str, int],
    limit: Optional[int],
    batch_size: int = 4,
    active_tasks: Optional[list[tuple]] = None,
) -> tuple[dict[str, float | None], dict[str, float]]:
    """Run benchmarks for *active_tasks* (defaults to the full TASKS list).

    Returns
    -------
    results : dict[label → score%]  — includes "Average" key
    elapsed : dict[label → seconds]
    """
    if active_tasks is None:
        active_tasks = TASKS
    results: dict[str, float | None] = {}
    elapsed: dict[str, float] = {}
    logger.info(f"Starting benchmark suite: {len(active_tasks)} tasks, device={device}, limit={limit}")

    for task_id, label, _, _, task_limit in active_tasks:
        # Step 1: Resolve per-task shot count and example limit.
        # n_shots is a per-task dict; tasks not listed default to 0-shot.
        # task_limit (from TASKS tuple) takes priority over the global limit so
        # expensive tasks like MMLU can be capped independently.
        # Example: task_id="gsm8k", n_shots={"gsm8k":8,"arc_challenge":25}
        #          n_shot=8, effective_limit=task_limit if set, else global limit
        n_shot = n_shots.get(task_id, 0)
        effective_limit = task_limit if task_limit is not None else limit
        n_examples = f"limit={effective_limit}" if effective_limit else "full"
        logger.info(f"--- Task: {label} ({task_id}) | {n_shot}-shot | {n_examples} ---")
        print(f"  [{label}] {task_id} | {n_shot}-shot | {n_examples} ...", flush=True)
        t0 = time.time()
        try:
            # Step 2: Call the task-specific eval function which returns a fraction in [0,1].
            # Convert to a percentage rounded to 4 decimal places for consistency across tasks.
            # Example: score=0.7543 (ARC) → results["ARC-Challenge"] = 75.43
            score = _EVAL_FNS[task_id](
                model, tokenizer, device,
                n_shot=n_shot,
                limit=effective_limit,
                batch_size=batch_size,
            )
            results[label] = round(score * 100.0, 4)
            elapsed[label] = round(time.time() - t0, 1)
            logger.info(f"  {label} complete: {results[label]:.2f}%  ({elapsed[label]:.0f}s)")
            print(f"    {label}: {results[label]:.2f}%  ({elapsed[label]:.0f}s)", flush=True)
        except Exception as exc:
            # Step 3: Catch task failures and record None so a single bad task doesn't
            # abort the entire benchmark suite. None is excluded from the Average.
            elapsed[label] = round(time.time() - t0, 1)
            logger.warning(f"Task {task_id} failed after {elapsed[label]:.0f}s: {exc}")
            results[label] = None

    # Step 4: Compute the macro-average across all tasks that returned a valid score.
    # None entries (failed tasks) are filtered out before computing the mean so a failure
    # doesn't pull the average toward 0.
    # Example: results={"ARC":75.43, "HellaSwag":None, "MMLU":62.10}
    #          vals=[75.43, 62.10] → Average = mean([75.43, 62.10]) = 68.77
    vals = [v for v in results.values() if v is not None]
    results["Average"] = round(mean(vals), 4) if vals else None
    logger.info(f"Benchmark suite complete. Average: {results['Average']}")
    return results, elapsed


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def delta(curr: dict, prev: dict) -> dict:
    """Compute per-task score deltas between two result dicts."""
    out: dict[str, float | None] = {}
    for k in curr:
        cv, pv = curr.get(k), prev.get(k)
        out[k] = None if (cv is None or pv is None) else round(cv - pv, 4)
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v: float | None) -> str:
    return "NA" if v is None else f"{v:.2f}"


def fmt_delta(v: float | None) -> str:
    if v is None:
        return "  NA  "
    arrow = "▲" if v > 0 else ("▼" if v < 0 else "─")
    sign  = "+" if v > 0 else ""
    return f"{arrow}{sign}{v:.2f}"


def iter_num(name: str) -> int:
    if name == "base_model":
        return -1
    m = re.search(r"iter_(\d+)", name)
    return int(m.group(1)) if m else _ITER_NUM_SENTINEL


# ---------------------------------------------------------------------------
# CLI results table
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    """Print a formatted ASCII table of all iteration results to stdout."""
    col_labels = ["Iteration", "Arc", "TruthfulQA", "Winogrande", "GSM8k", "HellaSwag", "MMLU", "Avg%", "ΔAvg", "Status"]

    data: list[list[str]] = []
    for row in rows:
        m  = row["metrics"]
        dp = row["delta_prev"]
        is_best   = row["best_iteration_so_far"] == row["iteration"]
        delta_avg = fmt_delta(dp.get("Average") if dp else None)
        data.append([
            row["iteration"],
            fmt(m.get("Arc")),        fmt(m.get("TruthfulQA")),
            fmt(m.get("Winogrande")), fmt(m.get("GSM8k")),
            fmt(m.get("HellaSwag")),  fmt(m.get("MMLU")),
            fmt(m.get("Average")),    delta_avg,
            "★ BEST" if is_best else "",
        ])

    widths = [max(len(col_labels[i]), max((len(r[i]) for r in data), default=0)) for i in range(len(col_labels))]
    sep        = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_row = "| " + " | ".join(col_labels[i].ljust(widths[i]) for i in range(len(col_labels))) + " |"

    print("\n" + sep)
    print(header_row)
    print(sep)
    for row_cells in data:
        print("| " + " | ".join(row_cells[i].ljust(widths[i]) for i in range(len(col_labels))) + " |")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(rows: list[dict], path: Path) -> None:
    """Write a human-readable comparative summary as an aligned table + narrative text file."""
    headers = [
        "Iteration", "Arc", "TruthfulQA", "Winogrande", "GSM8k",
        "HellaSwag", "MMLU", "Average", "DeltaPrevAvg", "BestSoFarAvg",
    ]

    data: list[list[str]] = []
    for row in rows:
        m, dp = row["metrics"], row["delta_prev"]
        data.append([
            row["iteration"],
            fmt(m.get("Arc")),        fmt(m.get("TruthfulQA")),
            fmt(m.get("Winogrande")), fmt(m.get("GSM8k")),
            fmt(m.get("HellaSwag")),  fmt(m.get("MMLU")),
            fmt(m.get("Average")),
            fmt(dp.get("Average") if dp else None),
            fmt(row.get("best_so_far_avg")),
        ])

    # Fixed-width columns so the file lines up in any text editor. Tabs do NOT align when
    # cell widths vary — that misalignment is what made the Average column look wrong even
    # though the values were correct. Mirrors print_results_table (the terminal output).
    widths = [max(len(headers[i]), max((len(r[i]) for r in data), default=0)) for i in range(len(headers))]
    sep        = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_row = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [sep, header_row, sep]
    for r in data:
        lines.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    lines.append(sep)

    lines += ["", "Per-iteration comparative analysis:"]
    for row in rows:
        lines.append(f"[{row['iteration']}]")
        m = row["metrics"]
        lines.append("scores: " + ", ".join(f"{k}={fmt(v)}" for k, v in m.items()))

        if row["delta_prev"]:
            d        = row["delta_prev"]
            improved = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v > 0]
            declined = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v < 0]
            stable   = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v == 0]
            lines.append("delta_vs_prev: " + ", ".join(f"{k}={fmt_delta(v)}" for k, v in d.items()))
            lines.append(f"improved_tasks={improved}  declined_tasks={declined}  stable_tasks={stable}")
        else:
            lines.append("delta_vs_prev: baseline iteration (no previous to compare against)")

        if row["best_iteration_so_far"] == row["iteration"]:
            lines.append("status: ★ NEW BEST average so far")
        else:
            lines.append(f"status: below best-so-far iteration {row['best_iteration_so_far']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparative summary written to: {path}")


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------

def _make_leaderboard_md(row: dict) -> str:
    """Render a single iteration's results as a Markdown table for TensorBoard."""
    m  = row["metrics"]
    dp = row["delta_prev"]
    lines = [
        f"## {row['iteration']} — Benchmark Scores",
        "",
        "| Task | Score | ΔPrev |",
        "|------|------:|------:|",
    ]
    for _, label, *_ in TASKS:
        if label not in m:
            continue
        score_str = fmt(m.get(label))
        delta_str = fmt_delta(dp.get(label) if dp else None)
        lines.append(f"| {label} | {score_str}% | {delta_str} |")
    lines.append(f"| **Average** | **{fmt(m.get('Average'))}%** | **{fmt_delta(dp.get('Average') if dp else None)}** |")
    lines.append("")
    if row["best_iteration_so_far"] == row["iteration"]:
        lines.append(f"**★ New best average: {fmt(m.get('Average'))}%**")
    else:
        lines.append(f"Best so far: {row['best_iteration_so_far']} ({fmt(row.get('best_so_far_avg'))}%)")
    return "\n".join(lines)


def _tb_setup(
    tb_dir: Path,
    checkpoints_dir: str,
    output_dir: str,
    device: str,
    limit,
    n_shots_resolved: dict,
    n_iters: int,
    active_tasks: list[tuple],
) -> "SummaryWriter":
    """Open a SummaryWriter, register the custom scalar layout, and write the run-config card."""
    logger.info(f"Opening TensorBoard writer: {tb_dir}")
    writer = SummaryWriter(log_dir=str(tb_dir))

    writer.add_custom_scalars({
        "Evaluation": {
            "Average Score (%)": ["Multiline", [_TB_TAG_AVG, _TB_TAG_BEST_AVG]],
            "Per-Task Scores":   ["Multiline", [f"{_TB_TAG_TASKS}/{t[1].lower()}" for t in active_tasks]],
        },
        "Delta vs Previous": {
            "Average Delta":      ["Multiline", [f"{_TB_TAG_DELTA_PREFIX}average"]],
            "Per-Task Deltas":    ["Multiline", [f"{_TB_TAG_DELTA_PREFIX}{t[1].lower()}" for t in active_tasks]],
            "Task Change Counts": ["Multiline", [_TB_TAG_IMPROVED_COUNT, _TB_TAG_DECLINED_COUNT]],
            "Improvement Rate":   ["Multiline", [_TB_TAG_IMPROVEMENT_RATE]],
        },
    })

    active_ids   = [t[0] for t in active_tasks]
    shot_summary = ", ".join(
        f"{k}={v}" for k, v in sorted(n_shots_resolved.items()) if k in active_ids
    )
    task_summary = ", ".join(f"{t[0]} ({t[1]})" for t in active_tasks)
    config_md = (
        "## Evaluation Run Configuration\n\n"
        f"| Setting | Value |\n"
        f"|---------|-------|\n"
        f"| Checkpoints dir | `{checkpoints_dir}` |\n"
        f"| Output dir | `{output_dir}` |\n"
        f"| Device | `{device}` |\n"
        f"| Example limit | `{limit if limit else 'full dataset'}` |\n"
        f"| Active tasks | `{task_summary}` |\n"
        f"| Shot counts | `{shot_summary}` |\n"
        f"| Iterations planned | `{n_iters}` |\n"
    )
    writer.add_text(_TB_TAG_RUN_CONFIG, config_md, global_step=0)
    return writer


def _tb_write_row(writer: "SummaryWriter", row: dict) -> None:
    """Write one iteration's metrics to an already-open SummaryWriter and flush."""
    step    = iter_num(row["iteration"])
    metrics = row["metrics"]

    avg = metrics.get("Average")
    if avg is not None:
        writer.add_scalar(_TB_TAG_AVG, avg, step)

    task_scores: dict[str, float] = {
        t[1].lower(): metrics[t[1]]
        for t in TASKS if metrics.get(t[1]) is not None
    }
    if task_scores:
        writer.add_scalars(_TB_TAG_TASKS, task_scores, step)
        for name, score in task_scores.items():
            writer.add_scalar(f"{_TB_TAG_TASKS}/{name}", score, step)

    writer.add_scalar(_TB_TAG_TASK_COUNT, len(task_scores), step)

    best_avg = row.get("best_so_far_avg")
    if best_avg is not None:
        writer.add_scalar(_TB_TAG_BEST_AVG, best_avg, step)

    if row["delta_prev"]:
        d = row["delta_prev"]
        for k, v in d.items():
            if v is not None:
                writer.add_scalar(f"{_TB_TAG_DELTA_PREFIX}{k.lower()}", v, step)

        improved_count = sum(1 for k, v in d.items() if k in _TASK_LABELS and v is not None and v > 0)
        declined_count = sum(1 for k, v in d.items() if k in _TASK_LABELS and v is not None and v < 0)
        total_valid    = sum(1 for k, v in d.items() if k in _TASK_LABELS and v is not None)
        writer.add_scalar(_TB_TAG_IMPROVED_COUNT, improved_count, step)
        writer.add_scalar(_TB_TAG_DECLINED_COUNT, declined_count, step)
        if total_valid > 0:
            writer.add_scalar(_TB_TAG_IMPROVEMENT_RATE, improved_count / total_valid, step)

    writer.add_text(_TB_TAG_SCORECARD, _make_leaderboard_md(row), step)

    if row["best_iteration_so_far"] == row["iteration"] and avg is not None:
        writer.add_text(
            _TB_TAG_BEST_ITER,
            f"**{row['iteration']}** achieved new best average: **{fmt(avg)}%**",
            step,
        )

    writer.flush()
    logger.info(f"TensorBoard: flushed metrics for {row['iteration']} (step={step})")


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    cfg: SPINConfig,
    iters: list[str] | None = None,
    active_tasks: list[tuple] | None = None,
    n_shots: dict[str, int] | None = None,
    include_base_model: bool = True,
) -> None:
    """Run benchmark evaluation for all (or specified) SPIN iteration checkpoints.

    Called from main.py after each SPIN iteration completes (when
    cfg.eval_run_after_training is True), or directly from the standalone main()
    below.  Each call auto-discovers every iter_* checkpoint under
    cfg.checkpoints_dir and skips any that already have a cached .parsed.json
    result, so repeated calls only evaluate the newest iteration.  Evaluation
    settings come from cfg (eval_output_dir, eval_batch_size, eval_limit, etc.);
    model loading and compilation reuse load_causal_lm() and maybe_compile_model()
    from utils.

    Parameters
    ----------
    cfg          : SPINConfig with all training + eval settings.
    iters        : Specific iteration names to evaluate (e.g. ["iter_0", "iter_2"]).
                   None = auto-discover all iter_* directories under cfg.checkpoints_dir.
    active_tasks : Subset of TASKS to run.  None = full TASKS list.
    n_shots      : Per-task shot count map.  None = DEFAULT_SHOTS from TASKS registry.
    """
    if active_tasks is None:
        active_tasks = list(TASKS)
    if n_shots is None:
        n_shots = dict(DEFAULT_SHOTS)

    logger.info("run_eval() starting")
    logger.info(f"  Checkpoints dir : {cfg.checkpoints_dir}")
    logger.info(f"  Output dir      : {cfg.eval_output_dir}")
    logger.info(f"  TensorBoard dir : {cfg.eval_tensorboard_dir}")
    logger.info(f"  Device          : {cfg.device}")
    logger.info(f"  Example limit   : {cfg.eval_limit if cfg.eval_limit else 'full dataset'}")
    logger.info(f"  Active tasks    : {[t[0] for t in active_tasks]}")
    logger.info(f"  Shot counts     : { {tid: n_shots[tid] for tid, *_ in active_tasks} }")

    ckpt_dir = Path(cfg.checkpoints_dir)
    out_dir  = Path(cfg.eval_output_dir)
    tb_dir   = Path(cfg.eval_tensorboard_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(tb_dir))

    if iters:
        iter_paths = [ckpt_dir / name for name in iters]
        logger.info(f"Evaluating specified iterations: {iters}")
    else:
        iter_paths = sorted(
            [p for p in ckpt_dir.iterdir() if p.is_dir() and re.match(r"iter_\d+", p.name)],
            key=lambda p: iter_num(p.name),
        )
        logger.info(f"Auto-discovered {len(iter_paths)} iteration(s) in {ckpt_dir}")

    if not iter_paths:
        raise SystemExit(f"No iteration directories found in {ckpt_dir}")

    logger.info(f"Iterations to evaluate: {[p.name for p in iter_paths]}")
    print(f"Found {len(iter_paths)} iteration(s) to evaluate: {[p.name for p in iter_paths]}")
    if cfg.eval_limit:
        logger.warning(f"SMOKE TEST: limit={cfg.eval_limit} examples per task — do not use for real benchmarks.")
        print(f"[SMOKE TEST] Limiting to {cfg.eval_limit} examples per task.")

    rows:         list[dict]   = []
    best_avg:     float | None = None
    best_iter:    str   | None = None
    prev_metrics: dict  | None = None

    run_start = time.time()
    tb_writer = _tb_setup(
        tb_dir,
        checkpoints_dir=str(ckpt_dir),
        output_dir=str(out_dir),
        device=cfg.device,
        limit=cfg.eval_limit,
        n_shots_resolved=n_shots,
        n_iters=len(iter_paths) + (1 if include_base_model else 0),
        active_tasks=active_tasks,
    )

    if include_base_model:
        base_name      = "base_model"
        base_json_path = out_dir / f"{base_name}.parsed.json"
        print(f"\n{'='*60}", flush=True)

        if base_json_path.exists() and not cfg.eval_no_cache:
            logger.info("========== Loading cached base_model ==========")
            print(f" base_model  →  [cached] {base_json_path}", flush=True)
            print(f"{'='*60}", flush=True)
            base_metrics              = json.loads(base_json_path.read_text())
            base_elapsed: dict[str, float] = {}
            base_ok = True
        else:
            logger.info("========== Evaluating base_model ==========")
            print(f" base_model  →  {cfg.model_name_or_path}", flush=True)
            print(f"{'='*60}", flush=True)
            base_ok = False
            base_metrics: dict       = {}
            base_elapsed             = {}
            try:
                load_cfg = dataclasses.replace(cfg, compile_ref_model=False)
                tokenizer = load_tokenizer(load_cfg)
                log_memory("before_load_base_model")
                model = load_causal_lm(cfg.model_name_or_path, load_cfg, trainable=False).to(cfg.device)
                log_memory("after_load_base_model")

                global MAX_SEQ_LEN
                model_max = getattr(model.config, "max_position_embeddings", MAX_SEQ_LEN)
                tok_max   = getattr(tokenizer, "model_max_length", MAX_SEQ_LEN)
                MAX_SEQ_LEN = min(cfg.eval_max_seq_len, model_max, tok_max)
                logger.info(f"  MAX_SEQ_LEN capped to {MAX_SEQ_LEN} (model={model_max}, tokenizer={tok_max})")

                if cfg.eval_compile_model:
                    compile_cfg = dataclasses.replace(cfg, compile_fullgraph=False, compile_mode="default")
                    model = maybe_compile_model(model, compile_cfg, label="eval_base_model")

                base_metrics, base_elapsed = run_all_benchmarks(
                    model, tokenizer, cfg.device, n_shots, cfg.eval_limit,
                    batch_size=cfg.eval_batch_size,
                    active_tasks=active_tasks,
                )
                log_memory("before_free_base_model")
                free_model(model)
                log_memory("after_free_base_model")
                save_json(str(base_json_path), base_metrics)
                logger.info(f"Base model scores saved to: {base_json_path}")
                base_ok = True
            except Exception as exc:
                logger.error(f"Failed to evaluate base model: {exc}")

        if base_ok:
            base_avg = base_metrics.get("Average")
            if base_avg is not None and (best_avg is None or base_avg > best_avg):
                best_avg  = base_avg
                best_iter = base_name

            base_row = {
                "iteration":             base_name,
                "metrics":               base_metrics,
                "delta_prev":            None,
                "best_so_far_avg":       best_avg,
                "best_iteration_so_far": best_iter,
                "elapsed_seconds":       base_elapsed,
            }
            rows.append(base_row)
            _tb_write_row(tb_writer, base_row)
            prev_metrics = base_metrics

            is_best_str = " ★ NEW BEST" if best_iter == base_name else ""
            total_time  = sum(base_elapsed.values())
            logger.info(
                f"base_model: avg={fmt(base_avg)}%  "
                f"best_so_far={fmt(best_avg)}%{is_best_str}  ({total_time:.0f}s)"
            )
            print(
                f"\n  [base_model]  avg={fmt(base_avg)}%  "
                f"(baseline — no delta)  "
                f"best_so_far={fmt(best_avg)}%{is_best_str}  "
                f"({total_time:.0f}s total)",
                flush=True,
            )

    for iter_path in iter_paths:
        iter_name = iter_path.name
        json_path = out_dir / f"{iter_name}.parsed.json"
        print(f"\n{'='*60}", flush=True)

        if json_path.exists() and not cfg.eval_no_cache:
            logger.info(f"========== Loading cached {iter_name} ==========")
            print(f" {iter_name}  →  [cached] {json_path}", flush=True)
            print(f"{'='*60}", flush=True)
            metrics         = json.loads(json_path.read_text())
            elapsed: dict[str, float] = {}
        else:
            model_path = find_model_path(iter_path)
            logger.info(f"========== Evaluating {iter_name} ==========")
            logger.info(f"  Model path: {model_path}")
            print(f" {iter_name}  →  {model_path}", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                # Override tokenizer path to load from the checkpoint, not cfg.model_name_or_path.
                # Disable compile_ref_model so we control compilation ourselves below.
                load_cfg = dataclasses.replace(
                    cfg,
                    tokenizer_name_or_path=str(model_path),
                    compile_ref_model=False,
                )
                tokenizer = load_tokenizer(load_cfg)
                log_memory(f"before_load_{iter_name}")
                model = load_causal_lm(str(model_path), load_cfg, trainable=False).to(cfg.device)
                log_memory(f"after_load_{iter_name}")

                # Cap MAX_SEQ_LEN to what this model can actually accept so the
                # truncation guards in score_continuations_batched / score_examples_batched
                # fire before PyTorch hits an out-of-bounds positional embedding.
                model_max = getattr(model.config, "max_position_embeddings", MAX_SEQ_LEN)
                tok_max   = getattr(tokenizer, "model_max_length", MAX_SEQ_LEN)
                MAX_SEQ_LEN = min(cfg.eval_max_seq_len, model_max, tok_max)
                logger.info(f"  MAX_SEQ_LEN capped to {MAX_SEQ_LEN} (model={model_max}, tokenizer={tok_max})")

                if cfg.eval_compile_model:
                    # eval uses model.generate (GSM8k), so fullgraph must be False.
                    compile_cfg = dataclasses.replace(cfg, compile_fullgraph=False, compile_mode="default")
                    model = maybe_compile_model(model, compile_cfg, label=f"eval_{iter_name}")
            except Exception as exc:
                logger.error(f"Failed to load model for {iter_name}: {exc}")
                continue

            metrics, elapsed = run_all_benchmarks(
                model, tokenizer, cfg.device, n_shots, cfg.eval_limit,
                batch_size=cfg.eval_batch_size,
                active_tasks=active_tasks,
            )

            log_memory(f"before_free_{iter_name}")
            free_model(model)
            log_memory(f"after_free_{iter_name}")

            save_json(str(json_path), metrics)
            logger.info(f"Scores saved to: {json_path}")

        dprev = delta(metrics, prev_metrics) if prev_metrics is not None else None

        avg = metrics["Average"]
        if avg is not None and (best_avg is None or avg > best_avg):
            best_avg  = avg
            best_iter = iter_name
            logger.info(f"New best average: {best_avg:.2f}% at {best_iter}")

        row = {
            "iteration":             iter_name,
            "metrics":               metrics,
            "delta_prev":            dprev,
            "best_so_far_avg":       best_avg,
            "best_iteration_so_far": best_iter,
            "elapsed_seconds":       elapsed,
        }
        rows.append(row)
        _tb_write_row(tb_writer, row)
        prev_metrics = metrics

        dprev_str   = fmt_delta(dprev.get("Average") if dprev else None)
        is_best_str = " ★ NEW BEST" if best_iter == iter_name else ""
        total_time  = sum(elapsed.values())
        logger.info(
            f"{iter_name}: avg={fmt(avg)}%  delta_prev={dprev_str}  "
            f"best_so_far={fmt(best_avg)}%{is_best_str}  ({total_time:.0f}s)"
        )
        print(
            f"\n  [{iter_name}]  avg={fmt(avg)}%  "
            f"Δprev_avg={dprev_str}  "
            f"best_so_far={fmt(best_avg)}%{is_best_str}  "
            f"({total_time:.0f}s total)",
            flush=True,
        )
        if dprev:
            improved = [k for k, v in dprev.items() if k in _TASK_LABELS and v is not None and v > 0]
            declined = [k for k, v in dprev.items() if k in _TASK_LABELS and v is not None and v < 0]
            if improved:
                logger.info(f"  Improved tasks: {improved}")
                print(f"  ▲ improved: {', '.join(improved)}", flush=True)
            if declined:
                logger.info(f"  Declined tasks: {declined}")
                print(f"  ▼ declined: {', '.join(declined)}", flush=True)

    if not rows:
        raise SystemExit("No successful evaluation results found.")

    print_results_table(rows)

    total_wall = time.time() - run_start
    logger.info(f"All evaluations complete. Total wall-clock time: {total_wall/60:.1f} min")
    print(f"Total evaluation time: {total_wall/60:.1f} min", flush=True)

    summary_path = out_dir / "comparative_summary.txt"
    write_summary(rows, summary_path)
    json_summary = out_dir / "comparative_summary.json"
    save_json(str(json_summary), rows)
    logger.info(f"JSON summary written to: {json_summary}")
    tb_writer.close()

    logger.info(f"Best iteration overall: {best_iter} ({fmt(best_avg)}%)")
    print(f"\nSaved summary : {summary_path}")
    print(f"TensorBoard   : {tb_dir}")
    print(f"  python -m tensorboard.main --logdir={tb_dir.parent}")
    print(f"\nBest iteration: {best_iter} ({fmt(best_avg)}%)")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _defaults = SPINConfig()
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate successive SPIN checkpoints on standard benchmarks, compare "
            "iteration-over-iteration, and log metrics to TensorBoard."
        )
    )
    ap.add_argument(
        "--checkpoints-dir", default=_defaults.checkpoints_dir,
        help="Root directory containing iter_* sub-directories.",
    )
    ap.add_argument(
        "--iters", nargs="*", default=None,
        help="Optional subset of iteration names to evaluate, e.g. iter_0 iter_1.",
    )
    ap.add_argument(
        "--output-dir", default=_defaults.eval_output_dir,
        help="Directory for per-iteration JSON files and the comparative summary.",
    )
    ap.add_argument(
        "--tensorboard-dir", default=_defaults.eval_tensorboard_dir,
        help="TensorBoard log directory for evaluation metrics.",
    )
    ap.add_argument("--device", default=_defaults.device)
    ap.add_argument(
        "--attn-impl", default=_defaults.attn_implementation or "sdpa",
        help="Attention implementation: sdpa (default), flash_attention_2, or empty string for eager.",
    )
    ap.add_argument(
        "--compile", action="store_true", default=_defaults.compile_model,
        help="Apply torch.compile() to each checkpoint model before evaluation (requires PyTorch >= 2.0).",
    )
    ap.add_argument(
        "--compile-backend", default=_defaults.compile_backend,
        help="torch.compile backend (default: inductor). Use aot_eager if triton is unavailable.",
    )
    ap.add_argument(
        "--limit", type=int, default=_defaults.eval_limit,
        help="Max examples per task for smoke testing. Do NOT use for real benchmarks.",
    )
    ap.add_argument(
        "--n-shots", nargs="*", default=None,
        help="Per-task shot count overrides: arc_challenge=10 gsm8k=3 (etc.).",
    )
    ap.add_argument(
        "--eval-batch-size", type=int, default=_defaults.eval_batch_size,
        help="Number of (context, continuation) rows per GPU forward pass. "
             "Increase for shorter sequences or larger GPUs; decrease if OOM.",
    )
    ap.add_argument(
        "--no-cache", action="store_true", default=_defaults.eval_no_cache,
        help="Re-evaluate iterations even if a cached JSON result already exists.",
    )

    valid_ids    = [t[0] for t in TASKS]
    valid_labels = [t[1] for t in TASKS]
    task_help = (
        "Task IDs or labels to run, space-separated. "
        f"Valid: {', '.join(f'{i}/{l}' for i, l in zip(valid_ids, valid_labels))}. "
        "Case-insensitive. Cannot be combined with --skip-tasks."
    )
    skip_help = (
        "Task IDs or labels to exclude, space-separated. "
        "All other tasks run. Cannot be combined with --tasks."
    )
    ap.add_argument("--tasks",      nargs="+", default=None, metavar="TASK", help=task_help)
    ap.add_argument("--skip-tasks", nargs="+", default=None, metavar="TASK", help=skip_help)
    ap.add_argument(
        "--no-base-model", action="store_true", default=False,
        help="Skip evaluating the base model before iteration checkpoints. "
             "By default the base model is evaluated first and used as the delta reference.",
    )

    args = ap.parse_args()

    if args.tasks and args.skip_tasks:
        raise SystemExit("--tasks and --skip-tasks are mutually exclusive.")

    if args.tasks:
        active_tasks = resolve_task_filter(args.tasks)
    elif args.skip_tasks:
        skip_ids     = {t[0] for t in resolve_task_filter(args.skip_tasks)}
        active_tasks = [t for t in TASKS if t[0] not in skip_ids]
    else:
        active_tasks = list(TASKS)

    if not active_tasks:
        raise SystemExit("No tasks selected — check --tasks / --skip-tasks arguments.")

    n_shots = dict(DEFAULT_SHOTS)
    if args.n_shots:
        for item in args.n_shots:
            k, _, v = item.partition("=")
            n_shots[k.strip()] = int(v.strip())

    cfg = dataclasses.replace(
        _defaults,
        checkpoints_dir  = args.checkpoints_dir,
        eval_output_dir  = args.output_dir,
        eval_tensorboard_dir = args.tensorboard_dir,
        device           = args.device,
        attn_implementation  = args.attn_impl or None,
        compile_model    = args.compile,
        compile_backend  = args.compile_backend,
        eval_limit       = args.limit,
        eval_batch_size  = args.eval_batch_size,
        eval_no_cache    = args.no_cache,
    )

    run_eval(
        cfg,
        iters=args.iters,
        active_tasks=active_tasks,
        n_shots=n_shots,
        include_base_model=not args.no_base_model,
    )


if __name__ == "__main__":
    main()
