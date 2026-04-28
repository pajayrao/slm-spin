"""
evaluate.py — Post-training benchmark evaluation and TensorBoard logging for SPIN.

Evaluates six standard LLM benchmarks directly using HuggingFace transformers —
no lm_eval dependency. Multiple-choice tasks (ARC, TruthfulQA-MC2, Winogrande,
HellaSwag, MMLU) use log-likelihood continuation scoring; GSM8k uses greedy
generation with answer extraction.

Flow overview
-------------
  1. Discover iteration checkpoint directories (iter_0, iter_1, …) under --checkpoints-dir.
  2. For each iteration:
       a. Load model + tokenizer from the checkpoint.
       b. Run all six benchmarks, collecting one scalar metric per task.
       c. Compute delta vs the previous iteration's scores.
       d. Track the running best-average across all evaluated iterations.
       e. Save per-iteration results to a JSON file.
       f. Free the model from GPU memory before loading the next checkpoint.
  3. Write a human-readable comparative summary (TXT + JSON).
  4. Write TensorBoard scalars, text cards, and metadata for the full run.

Usage
-----
  python evaluate.py
  python evaluate.py --iters iter_0 iter_2
  python evaluate.py --limit 50          # smoke test — subset of examples
  python evaluate.py --n-shots arc_challenge=10 gsm8k=3
"""

import argparse
import logging
import os
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

# Ordered list of sub-directory names probed inside each iter_* checkpoint
# directory when looking for a HuggingFace model root.  "." means the iter
# directory itself.  Order matters: more specific names are checked first.
_CHECKPOINT_SUBDIR_CANDIDATES: list[str] = [
    "hf_final", "final_checkpoint", "checkpoint-final", "merged", "model", ".",
]

# Files whose presence in a directory marks it as a valid HuggingFace model root.
# adapter_config.json covers LoRA checkpoints that were not yet merged.
_MODEL_CONFIG_MARKERS: list[str] = ["config.json", "adapter_config.json"]

# Whether to allow the Hub to execute custom modelling code bundled with
# the checkpoint.  Set True only for explicitly trusted model repositories.
_TRUST_REMOTE_CODE: bool = False

# Whether to allow custom dataset loading scripts from the Hub.
# Public benchmark datasets (ARC, TruthfulQA, etc.) are trusted by convention.
_TRUST_REMOTE_CODE_DATASETS: bool = True

# ---------------------------------------------------------------------------
# Generation (GSM8k)
# ---------------------------------------------------------------------------

# Maximum new tokens the model may generate per GSM8k answer.
# 256 covers multi-step CoT chains and matches the lm_eval default.
GSM8K_MAX_NEW_TOKENS: int = 256

# ---------------------------------------------------------------------------
# Default CLI argument values
# ---------------------------------------------------------------------------

# These mirror the SPIN training output layout so evaluate.py works out of the
# box after a training run without any extra flags.
DEFAULT_BASE_DIR = "./spin_outputs_1"
DEFAULT_CHECKPOINTS_DIR = os.path.join(DEFAULT_BASE_DIR, "checkpoints")
DEFAULT_OUTPUT_DIR      = os.path.join(DEFAULT_BASE_DIR, "eval_results")
DEFAULT_TENSORBOARD_DIR = os.path.join(DEFAULT_BASE_DIR, "tensorboard/eval_compare")
DEFAULT_DEVICE          = "cuda"

# ---------------------------------------------------------------------------
# TensorBoard tag strings
# ---------------------------------------------------------------------------

# Centralised here so a tag rename touches one line rather than every call site.
_TB_TAG_AVG              = "eval/average"
_TB_TAG_BEST_AVG         = "eval/best_so_far_average"
_TB_TAG_TASK_COUNT       = "eval/task_count"
_TB_TAG_TASKS            = "eval/tasks"          # prefix for grouped + individual task scalars
_TB_TAG_SCORECARD        = "eval/scorecard"
_TB_TAG_BEST_ITER        = "eval/best_iteration"
_TB_TAG_RUN_CONFIG       = "eval/run_config"
_TB_TAG_DELTA_PREFIX     = "compare_vs_prev/"    # per-task delta tags are built from this prefix
_TB_TAG_IMPROVED_COUNT   = "compare_vs_prev/improved_task_count"
_TB_TAG_DECLINED_COUNT   = "compare_vs_prev/declined_task_count"
_TB_TAG_IMPROVEMENT_RATE = "compare_vs_prev/improvement_rate"

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

# Returned by iter_num() when a directory name has no numeric index, ensuring
# non-standard directories sort after all iter_* directories.
_ITER_NUM_SENTINEL: int = 10**9


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
# Each entry is (task_id, display_label, metric_type).
#   task_id      — key into _EVAL_FNS and DEFAULT_SHOTS
#   display_label— human-readable name used in tables and TensorBoard tags
#   metric_type  — how the score is computed:
#                    "acc"      = raw accuracy (fraction correct)
#                    "acc_norm" = length-normalised accuracy (score ÷ char count before argmax)
#                    "mc2"      = probability mass on all correct answers (TruthfulQA-specific)
TASKS = [
    ("arc_challenge",  "Arc",        "acc_norm"),
    ("truthfulqa_mc2", "TruthfulQA", "mc2"),
    ("winogrande",     "Winogrande", "acc"),
    ("gsm8k",          "GSM8k",      "acc"),
    ("hellaswag",      "HellaSwag",  "acc_norm"),
    ("mmlu",           "MMLU",       "acc"),
]

# Standard few-shot counts that match the Open LLM Leaderboard v1 setup.
# These are the community-standard shot counts — changing them would make
# scores incomparable to published results.
DEFAULT_SHOTS: dict[str, int] = {
    "arc_challenge":   25,
    "truthfulqa_mc2":   0,   # TruthfulQA is always evaluated zero-shot
    "winogrande":       5,
    "gsm8k":            5,
    "hellaswag":       10,
    "mmlu":             5,
}

# Per-dataset row limits applied before the global --limit flag.
# Set to an integer to cap the number of rows used from that dataset;
# None means fall back to the global --limit (or all rows if --limit is also None).
DATASET_LIMITS: dict[str, Optional[int]] = {
    "arc_challenge":  100,
    "truthfulqa_mc2": 100,
    "winogrande":     100,
    "gsm8k":          100,
    "hellaswag":      100,
    "mmlu":           100,
}

# Quick lookup: is a given key a task label (vs "Average")?
_TASK_LABELS: set[str] = {label for _, label, _ in TASKS}

# Maximum total token length (context + continuation) fed to the model.
# Sequences longer than this are truncated from the *left* so that the
# continuation tokens — which carry the gradient signal — are always present.
MAX_SEQ_LEN = 2048

# Standard 5-shot Chain-of-Thought examples used by the GSM8k paper and lm_eval.
# These are fixed exemplars — they are prepended verbatim to every test question.
# The "#### <number>" suffix is the canonical answer format the model must produce.
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
    """Locate the HuggingFace model directory inside an iteration checkpoint folder.

    SPIN saves checkpoints in varying sub-directory layouts depending on whether
    LoRA merging happened and what save_strategy was used.  This function probes a
    prioritised list of candidate sub-directories and falls back to a recursive
    glob for config.json / adapter_config.json if none of the candidates match.

    Returns the resolved absolute path to the directory that contains the model
    config — this path is suitable to pass directly to AutoModelForCausalLM.from_pretrained().
    """
    # Ordered by likelihood: hf_final is written by our save flow; the others
    # cover common HuggingFace trainer conventions.
    logger.info(f"Searching for model root in: {iter_dir}")
    candidates = _CHECKPOINT_SUBDIR_CANDIDATES
    for cand in candidates:
        p = iter_dir / cand
        if not p.is_dir():
            continue
        if any((p / m).exists() for m in _MODEL_CONFIG_MARKERS):
            logger.info(f"  Found model root at candidate '{cand}': {p.resolve()}")
            return p.resolve()
        # "." means iter_dir itself; config.json might sit directly there
        if cand == "." and any((iter_dir / m).exists() for m in _MODEL_CONFIG_MARKERS):
            logger.info(f"  Found model root at iteration directory itself: {iter_dir.resolve()}")
            return iter_dir.resolve()

    # Last-resort: walk the entire subtree to find a config marker
    logger.info("  No candidate matched — falling back to recursive search.")
    for marker in _MODEL_CONFIG_MARKERS:
        found = list(iter_dir.rglob(marker))
        if found:
            logger.info(f"  Recursive search found {marker} at: {found[0].parent.resolve()}")
            return found[0].parent.resolve()

    # Give up — return iter_dir and let from_pretrained raise a meaningful error
    logger.warning(f"  Could not locate a model config under {iter_dir}; will try iter_dir directly.")
    return iter_dir.resolve()


def load_model_and_tokenizer(model_path: str, device: str):
    """Load a causal LM and its tokenizer from a local checkpoint directory.

    Uses bfloat16 on CUDA (half memory, same dynamic range as float32) and
    float32 on CPU.  Sets pad_token = eos_token when no pad token is defined —
    this is standard practice and does not affect generation quality because
    padding tokens are masked during attention.
    """
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=_TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  pad_token was None — set to eos_token.")

    # bfloat16 is preferred on CUDA: it halves memory vs float32 and avoids the
    # narrow dynamic range of float16 that can cause NaN in large-scale models.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Loading model: dtype={torch_dtype}, device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        trust_remote_code=_TRUST_REMOTE_CODE,
    ).to(device)
    model.eval()   # disable dropout; we are only doing inference here
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model loaded: {n_params:.0f}M parameters, eval mode.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core log-likelihood scoring primitive
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: str,
) -> float:
    """Compute sum of log P(token | context) over all tokens in *continuation*.

    This is the standard approach used by lm_eval for multiple-choice scoring:
    the choice with the highest total log-likelihood (or highest per-character
    log-likelihood for acc_norm) is selected as the model's answer.

    Why tokenize together (context + continuation)?
    -----------------------------------------------
    Tokenizers can split the boundary between two strings differently when they
    are tokenized separately vs concatenated (the "boundary artifact" problem).
    For example, " dog" at the end of a context vs at the start of a continuation
    may produce different sub-word tokens.  Tokenizing the full string together
    ensures the tokens the model sees during scoring exactly match what they would
    be in free-form generation.

    Truncation strategy
    -------------------
    When the combined sequence exceeds MAX_SEQ_LEN, we truncate from the *left*
    of the context window.  This preserves the continuation tokens intact, which
    are the tokens we are actually scoring.  We always keep at least one context
    token before the continuation so the model has something to condition on.

    Returns
    -------
    float — total log-probability of the continuation tokens.  Higher is better
            (less negative).  Typical range: -200 to 0 for a short phrase.
    """
    # Tokenize the full string together to avoid boundary artifacts
    full_ids: list[int] = tokenizer(
        context + continuation, add_special_tokens=True
    )["input_ids"]

    # Determine where the continuation starts by measuring its standalone length.
    # We deliberately do NOT add special tokens here because the special token
    # count is already accounted for in full_ids.
    cont_ids_raw: list[int] = tokenizer(
        continuation, add_special_tokens=False
    )["input_ids"]
    cont_len = len(cont_ids_raw)

    if cont_len == 0:
        return 0.0

    # The continuation starts at full_ids[-cont_len]
    cont_start = len(full_ids) - cont_len

    # Truncate from the left to fit within the model's context window
    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[-(MAX_SEQ_LEN):]
        cont_start = len(full_ids) - cont_len
        if cont_start < 1:
            cont_start = 1  # must have at least one context token before the first predicted token

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # model() returns logits of shape [batch=1, seq_len, vocab_size].
    # logits[i] is the distribution over the *next* token given all tokens up to position i.
    logits = model(input_ids=input_ids).logits[0]  # [seq, vocab]
    log_probs = F.log_softmax(logits, dim=-1)

    # logits[cont_start - 1] predicts full_ids[cont_start], i.e. the first continuation token.
    # We slice log_probs to get one row per continuation token and gather the log-prob of
    # the actual token id at each position.
    cont_token_ids = torch.tensor(full_ids[cont_start:], dtype=torch.long, device=device)
    pred_log_probs = log_probs[cont_start - 1 : cont_start - 1 + cont_len]
    token_log_probs = pred_log_probs[torch.arange(cont_len, device=device), cont_token_ids]

    # Sum gives the joint log-probability of the entire continuation
    return token_log_probs.sum().item()


def _pick_best(scores: list[float], choices: list[str], normalize: bool) -> int:
    """Select the index of the highest-scoring choice.

    When normalize=True (acc_norm metric) each score is divided by the character
    length of the choice text before comparison.  This prevents the model from
    trivially preferring shorter answers just because they have fewer tokens to
    accumulate negative log-probabilities over.
    """
    if normalize:
        adjusted = [s / max(len(c), 1) for s, c in zip(scores, choices)]
    else:
        adjusted = list(scores)
    return int(max(range(len(adjusted)), key=lambda i: adjusted[i]))


# ---------------------------------------------------------------------------
# ARC Challenge  (metric: acc_norm)
# ---------------------------------------------------------------------------

def eval_arc_challenge(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate ARC-Challenge (Clark et al., 2018) using length-normalised log-likelihood.

    ARC-Challenge is a set of grade-school science multiple-choice questions selected
    because retrieval-based and word-co-occurrence methods all fail on them.

    Metric: acc_norm — the model is correct if the length-normalised log-likelihood
    (log P / character count) is highest for the correct answer choice.  Normalising
    by length prevents bias toward shorter choices.

    Few-shot format:
        Question: <question text>
        Answer: <correct answer text>

    The answer text (not the A/B/C/D label) is used both in few-shot exemplars
    and as the scored continuation for each test choice.
    """
    logger.info(f"ARC-Challenge: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  ARC-Challenge: {len(test_examples)} test examples.")

    # Build the few-shot prefix from training examples.
    # Each exemplar appends the full answer text, not just the letter.
    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            # Find the text corresponding to the correct answer label
            answer_text = texts[labels.index(ex["answerKey"])]
            few_shot_prefix += f"Question: {ex['question']}\nAnswer: {answer_text}\n\n"

    correct = 0
    logger.info("  ARC-Challenge: scoring examples...")
    for ex in test_examples:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        # Context ends with "Answer:" — the model's job is to complete it
        context = few_shot_prefix + f"Question: {ex['question']}\nAnswer:"
        # Each choice is prepended with a space for proper tokenisation boundary
        choices = [f" {t}" for t in texts]
        scores = [score_continuation(model, tokenizer, context, c, device) for c in choices]
        predicted_idx = _pick_best(scores, choices, normalize=True)
        if labels[predicted_idx] == ex["answerKey"]:
            correct += 1

    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# TruthfulQA MC2  (metric: mc2)
# ---------------------------------------------------------------------------

def eval_truthfulqa_mc2(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate TruthfulQA using the MC2 (multiple-correct) scoring metric.

    TruthfulQA measures whether a model outputs truthful statements.  The MC2
    variant has *multiple* correct answers per question (e.g. several true
    paraphrases), which makes it harder to game than a single-correct-answer setup.

    Metric: mc2 — for each question, compute softmax over all choice log-likelihoods
    to get a probability distribution, then sum the probability mass assigned to all
    correct choices.  A score of 1.0 means all probability mass went to true answers.

    Why softmax over log-likelihoods?
    ----------------------------------
    We want probabilities that sum to 1 across all choices so that we can measure
    how much mass the model places on the set of correct answers.  Taking softmax of
    the raw log-likelihoods is the standard lm_eval approach for mc2.

    TruthfulQA is evaluated zero-shot (n_shot=0) because there is no training split
    with reliable few-shot exemplars.
    """
    logger.info(f"TruthfulQA MC2: loading dataset (zero-shot, limit={limit})")
    if n_shot != 0:
        logger.warning(f"TruthfulQA has no few-shot train split; n_shot={n_shot} ignored.")
    ds = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    examples = list(ds["validation"])
    if limit:
        examples = examples[:limit]
    logger.info(f"  TruthfulQA MC2: {len(examples)} validation examples.")

    mc2_scores: list[float] = []
    for ex in examples:
        choices = ex["mc2_targets"]["choices"]
        labels = ex["mc2_targets"]["labels"]  # 1 = correct answer, 0 = incorrect
        context = f"Q: {ex['question']}\nA:"

        # Compute log-likelihood of each choice continuation, then softmax to get probs
        log_lls = torch.tensor(
            [score_continuation(model, tokenizer, context, f" {c}", device) for c in choices],
            dtype=torch.float64,
        )
        # Softmax turns the log-likelihoods into a valid probability distribution
        probs = torch.softmax(log_lls, dim=0)

        # mc2 score = total probability mass assigned to the correct choices
        correct_prob = float(sum(probs[i] for i, lbl in enumerate(labels) if lbl == 1))
        mc2_scores.append(correct_prob)

    # Average mc2 score across all questions — higher is more truthful
    return float(sum(mc2_scores) / len(mc2_scores))


# ---------------------------------------------------------------------------
# Winogrande  (metric: acc)
# ---------------------------------------------------------------------------

def eval_winogrande(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate Winogrande (Sakaguchi et al., 2019) on commonsense pronoun resolution.

    Winogrande is a large-scale Winograd Schema challenge.  Each question is a sentence
    with a blank (_) that can be filled by one of two options, testing commonsense reasoning
    about pronoun coreference.

    Metric: acc — the model is correct if the option with higher log-likelihood for
    (context_up_to_blank + option + rest_of_sentence) is the correct fill.

    Example:
        Sentence: "The trophy doesn't fit in the suitcase because _ is too large."
        Option1: "the trophy" (correct)
        Option2: "the suitcase"
        Context: "The trophy doesn't fit in the suitcase because "
        Scored continuations: "the trophy is too large." vs "the suitcase is too large."

    Few-shot exemplars present the filled sentence without explicit Q/A labelling,
    letting the model learn the pattern implicitly.
    """
    logger.info(f"Winogrande: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("winogrande", "winogrande_xl", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["validation"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  Winogrande: {len(test_examples)} validation examples.")

    # Build few-shot prefix: each exemplar is just the correctly-filled sentence
    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            answer_option = ex["option1"] if ex["answer"] == "1" else ex["option2"]
            few_shot_prefix += ex["sentence"].replace("_", answer_option) + "\n\n"

    correct = 0
    for ex in test_examples:
        sentence = ex["sentence"]
        blank_idx = sentence.index("_")
        # Context is everything before the blank; rest is everything after
        context = few_shot_prefix + sentence[:blank_idx]
        rest = sentence[blank_idx + 1:]  # text that follows the blank in the sentence

        # Score both options as continuations of the context, then pick the higher one
        s1 = score_continuation(model, tokenizer, context, ex["option1"] + rest, device)
        s2 = score_continuation(model, tokenizer, context, ex["option2"] + rest, device)
        predicted = "1" if s1 > s2 else "2"
        if predicted == ex["answer"]:
            correct += 1

    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# GSM8k  (metric: acc — exact numeric match)
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from a generated or reference response.

    Primary strategy: look for the GSM8k canonical delimiter "####" followed by a number.
    This matches the format used by both the few-shot exemplars and the ground truth.

    Fallback: return the last number found anywhere in the text.  This handles
    models that produce the correct answer without the delimiter.

    Commas in numbers (e.g. "1,234") are stripped before comparison so that
    "1234" and "1,234" compare as equal.
    """
    # Canonical GSM8k answer format: "#### 42" or "#### 1,234.5"
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last standalone number in the text
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = GSM8K_MAX_NEW_TOKENS) -> str:
    """Run greedy decoding to generate a response to a prompt.

    Unlike the log-likelihood tasks, GSM8k requires the model to *produce* a number
    rather than select among given options.  We use greedy (do_sample=False) decoding
    because the standard lm_eval GSM8k evaluation is deterministic.

    The prompt is truncated to MAX_SEQ_LEN - max_new_tokens to guarantee there is always
    room for the model to generate its answer before hitting the token limit.

    Returns the decoded text of the newly generated tokens only (prompt excluded).
    """
    enc = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN - max_new_tokens,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy: deterministic, matches lm_eval
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Strip the prompt tokens from the output; return only the newly generated text
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def eval_gsm8k(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate GSM8k (Cobbe et al., 2021) grade-school math via greedy generation.

    GSM8k tests multi-step arithmetic reasoning.  Unlike the multiple-choice tasks,
    the model must generate its answer freely — we then extract the final number from
    the generated text and compare it to the ground-truth number.

    Metric: acc — 1 if the extracted number string matches exactly, 0 otherwise.

    The 5 hard-coded Chain-of-Thought exemplars in _GSM8K_FEW_SHOT are the canonical
    set used by Wei et al. (2022) and lm_eval; they teach the model to show its work
    before writing "#### <answer>".
    """
    logger.info(f"GSM8k: loading dataset (n_shot={n_shot}, limit={limit}, max_new_tokens={GSM8K_MAX_NEW_TOKENS})")
    ds = load_dataset("gsm8k", "main", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  GSM8k: {len(test_examples)} test examples.")

    # Concatenate the CoT few-shot exemplars into a single prefix string
    few_shot_prefix = "".join(
        f"Question: {q}\nAnswer: {a}\n\n"
        for q, a in _GSM8K_FEW_SHOT[:n_shot]
    )

    correct = 0
    for ex in test_examples:
        prompt = few_shot_prefix + f"Question: {ex['question']}\nAnswer:"
        generated = _generate(model, tokenizer, prompt, device)
        # Both sides use _extract_number so the comparison is format-agnostic
        if _extract_number(generated) == _extract_number(ex["answer"]):
            correct += 1

    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# HellaSwag  (metric: acc_norm)
# ---------------------------------------------------------------------------

def _clean_hellaswag(text: str) -> str:
    """Strip annotation artifacts from HellaSwag text fields.

    Raw HellaSwag strings contain bracketed entity annotations like "[header]" or
    "[substeps]".  These are preprocessing artefacts not present in the original
    activity descriptions and would confuse the model.  We also collapse
    multiple consecutive whitespace characters into a single space.
    """
    text = re.sub(r"\[.*?\]", "", text)          # remove [bracket] annotations
    return re.sub(r"\s+", " ", text).strip()     # normalise whitespace


def eval_hellaswag(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate HellaSwag (Zellers et al., 2019) on commonsense sentence completion.

    HellaSwag provides an activity description and a partial sentence (ctx); the model
    must choose the most plausible of four possible endings.  Wrong endings are
    adversarially chosen to be misleading to n-gram and shallow language models.

    Metric: acc_norm — length-normalised log-likelihood over the four endings.
    Normalising removes the bias toward shorter endings.

    Context format: "<activity label>: <partial sentence context>"
    Continuation: " <cleaned ending text>"
    """
    logger.info(f"HellaSwag: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("Rowan/hellaswag", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    val_examples = list(ds["validation"])
    if limit:
        val_examples = val_examples[:limit]
    logger.info(f"  HellaSwag: {len(val_examples)} validation examples.")

    # Build few-shot prefix from training examples: just the context + correct ending
    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            ctx = _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"])
            best_ending = _clean_hellaswag(ex["endings"][int(ex["label"])])
            few_shot_prefix += f"{ctx} {best_ending}\n\n"

    correct = 0
    for ex in val_examples:
        context = few_shot_prefix + _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"])
        # Each ending is prepended with a space for consistent tokenisation
        endings = [" " + _clean_hellaswag(e) for e in ex["endings"]]
        scores = [score_continuation(model, tokenizer, context, e, device) for e in endings]
        predicted = _pick_best(scores, endings, normalize=True)
        if predicted == int(ex["label"]):
            correct += 1

    return correct / len(val_examples)


# ---------------------------------------------------------------------------
# MMLU  (metric: acc)
# ---------------------------------------------------------------------------

# MMLU uses A/B/C/D labels — the model must assign highest log P to the correct letter
_MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def _mmlu_format(ex: dict, with_answer: bool = False) -> str:
    """Render an MMLU example into the prompt format expected by the model.

    Format:
        The following is a multiple choice question about <subject>.
        <question>
        A. <choice0>
        B. <choice1>
        C. <choice2>
        D. <choice3>
        Answer:[ A]    ← only when with_answer=True (few-shot exemplars)

    When with_answer=False the prompt ends with "Answer:" and the model's job
    is to produce the correct letter as a one-token continuation.
    """
    subj = ex.get("subject", "").replace("_", " ")
    header = f"The following is a multiple choice question about {subj}.\n" if subj else ""
    choices_str = "\n".join(
        f"{lbl}. {ex['choices'][i]}" for i, lbl in enumerate(_MMLU_CHOICE_LABELS)
    )
    text = header + f"{ex['question'].strip()}\n{choices_str}\nAnswer:"
    if with_answer:
        # Append the letter and two blank lines to clearly separate from the next exemplar
        text += f" {_MMLU_CHOICE_LABELS[ex['answer']]}\n\n"
    return text


def eval_mmlu(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int]
) -> float:
    """Evaluate MMLU (Hendrycks et al., 2021) across 57 academic subjects.

    MMLU is a broad knowledge benchmark spanning humanities, STEM, social sciences,
    and professional domains.  Each question is 4-way multiple choice; the model
    must assign the highest log-likelihood to the single correct letter continuation.

    Metric: acc — no normalisation (all choices are single letters of the same length).

    Few-shot strategy: MMLU provides a dev split with 5 examples per subject.  We
    use per-subject few-shot prompts so the model sees relevant domain exemplars.
    If a subject has no dev examples, the question is evaluated zero-shot.
    """
    logger.info(f"MMLU: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("cais/mmlu", "all", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  MMLU: {len(test_examples)} test examples across all subjects.")

    # Pre-index dev examples by subject so we can quickly build per-question few-shot prefixes
    dev_by_subject: dict[str, list] = {}
    for ex in ds.get("dev", []):
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # The four possible continuations after "Answer:" — one token each
    choices = [f" {lbl}" for lbl in _MMLU_CHOICE_LABELS]
    correct = 0

    for ex in test_examples:
        # Use up to n_shot dev examples from the same subject as the test question
        few_shot_prefix = ""
        if n_shot > 0:
            for shot in dev_by_subject.get(ex.get("subject", ""), [])[:n_shot]:
                few_shot_prefix += _mmlu_format(shot, with_answer=True)

        context = few_shot_prefix + _mmlu_format(ex)
        # No length normalisation: all choice labels are single characters
        scores = [score_continuation(model, tokenizer, context, c, device) for c in choices]
        predicted = _pick_best(scores, choices, normalize=False)
        if predicted == ex["answer"]:
            correct += 1

    return correct / len(test_examples)


# ---------------------------------------------------------------------------
# Benchmark dispatcher
# ---------------------------------------------------------------------------

# Maps task_id → evaluation function; must stay in sync with TASKS
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
) -> tuple[dict[str, float | None], dict[str, float]]:
    """Run every benchmark in TASKS and return (results, elapsed_seconds_per_task).

    For each task:
      - Calls the appropriate eval function.
      - Converts the raw fraction (0.0–1.0) to a percentage (0.0–100.0).
      - Records elapsed wall-clock time in seconds.
      - Catches and logs exceptions so one failing task does not abort the run.

    Also computes the macro-average of all non-None task scores and adds it under
    the key "Average".  This matches the Open LLM Leaderboard v1 aggregation.

    Returns
    -------
    results : dict[label → score%]
        Includes an "Average" key.  Values are None for failed tasks.
    elapsed : dict[label → seconds]
        Wall-clock time spent on each task (useful for estimating full-run cost).
    """
    results: dict[str, float | None] = {}
    elapsed: dict[str, float] = {}
    logger.info(f"Starting benchmark suite: {len(TASKS)} tasks, device={device}, limit={limit}")

    for task_id, label, _ in TASKS:
        n_shot = n_shots.get(task_id, 0)
        # Per-dataset limit takes priority; fall back to the global CLI limit.
        dataset_limit = DATASET_LIMITS.get(task_id)
        effective_limit = dataset_limit if dataset_limit is not None else limit
        n_examples = f"limit={effective_limit}" if effective_limit else "full"
        logger.info(f"--- Task: {label} ({task_id}) | {n_shot}-shot | {n_examples} ---")
        print(
            f"  [{label}] {task_id} | {n_shot}-shot | {n_examples} ...",
            flush=True,
        )
        t0 = time.time()
        try:
            score = _EVAL_FNS[task_id](
                model, tokenizer, device,
                n_shot=n_shot,
                limit=effective_limit,
            )
            results[label] = round(score * 100.0, 4)
            elapsed[label] = round(time.time() - t0, 1)
            logger.info(f"  {label} complete: {results[label]:.2f}%  ({elapsed[label]:.0f}s)")
            print(
                f"    {label}: {results[label]:.2f}%  ({elapsed[label]:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            elapsed[label] = round(time.time() - t0, 1)
            logger.warning(f"Task {task_id} failed after {elapsed[label]:.0f}s: {exc}")
            results[label] = None

    # Macro-average over all tasks that successfully completed
    vals = [v for v in results.values() if v is not None]
    results["Average"] = round(mean(vals), 4) if vals else None
    logger.info(f"Benchmark suite complete. Average: {results['Average']}")
    return results, elapsed


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def delta(curr: dict, prev: dict) -> dict:
    """Compute per-task score deltas between two result dicts.

    Returns None for any task that failed in either iteration so that downstream
    code can clearly distinguish "no change" (0.0) from "one run failed" (None).
    """
    out: dict[str, float | None] = {}
    for k in curr:
        cv, pv = curr.get(k), prev.get(k)
        out[k] = None if (cv is None or pv is None) else round(cv - pv, 4)
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v: float | None) -> str:
    """Format a score for tabular display; returns 'NA' for missing values."""
    return "NA" if v is None else f"{v:.2f}"


def fmt_delta(v: float | None) -> str:
    """Format a delta with +/- sign and arrow indicator for CLI readability."""
    if v is None:
        return "  NA  "
    arrow = "▲" if v > 0 else ("▼" if v < 0 else "─")
    sign = "+" if v > 0 else ""
    return f"{arrow}{sign}{v:.2f}"


def iter_num(name: str) -> int:
    """Extract the numeric index from an iteration directory name like 'iter_3'.

    Returns a large sentinel value when no number is found so that non-standard
    directories sort after all iter_* directories.
    """
    m = re.search(r"iter_(\d+)", name)
    return int(m.group(1)) if m else _ITER_NUM_SENTINEL


# ---------------------------------------------------------------------------
# CLI results table
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    """Print a formatted ASCII table of all iteration results to stdout.

    Columns: Iteration | Arc | TruthfulQA | Winogrande | GSM8k | HellaSwag | MMLU | Average | ΔAvg | Status
    Each score is shown as XX.XX%; delta cells show ▲/▼ indicators.
    The best-average iteration is annotated with ★.
    """
    col_labels = ["Iteration", "Arc", "TruthfulQA", "Winogrande", "GSM8k", "HellaSwag", "MMLU", "Avg%", "ΔAvg", "Status"]

    # Build all data rows first so we can compute column widths
    data: list[list[str]] = []
    for row in rows:
        m = row["metrics"]
        dp = row["delta_prev"]
        is_best = row["best_iteration_so_far"] == row["iteration"]
        status = "★ BEST" if is_best else ""
        delta_avg = fmt_delta(dp.get("Average") if dp else None)
        data.append([
            row["iteration"],
            fmt(m.get("Arc")),
            fmt(m.get("TruthfulQA")),
            fmt(m.get("Winogrande")),
            fmt(m.get("GSM8k")),
            fmt(m.get("HellaSwag")),
            fmt(m.get("MMLU")),
            fmt(m.get("Average")),
            delta_avg,
            status,
        ])

    # Column widths: max of header and all data cells
    widths = [max(len(col_labels[i]), max((len(r[i]) for r in data), default=0)) for i in range(len(col_labels))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
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
    """Write a human-readable comparative summary to a TSV + narrative text file.

    The file has two sections:
      1. TSV table — paste into a spreadsheet for plotting.
      2. Per-iteration narrative blocks — improved/declined/stable task lists
         and a best-so-far annotation for quick scanning.
    """
    headers = [
        "Iteration", "Arc", "TruthfulQA", "Winogrande", "GSM8k",
        "HellaSwag", "MMLU", "Average", "DeltaPrevAvg", "BestSoFarAvg",
    ]
    lines = ["\t".join(headers)]

    for row in rows:
        m, dp = row["metrics"], row["delta_prev"]
        vals = [
            row["iteration"],
            fmt(m.get("Arc")),        fmt(m.get("TruthfulQA")),
            fmt(m.get("Winogrande")), fmt(m.get("GSM8k")),
            fmt(m.get("HellaSwag")),  fmt(m.get("MMLU")),
            fmt(m.get("Average")),
            fmt(dp.get("Average") if dp else None),
            fmt(row.get("best_so_far_avg")),
        ]
        lines.append("\t".join(vals))

    lines += ["", "Per-iteration comparative analysis:"]

    for row in rows:
        lines.append(f"[{row['iteration']}]")
        m = row["metrics"]
        lines.append("scores: " + ", ".join(f"{k}={fmt(v)}" for k, v in m.items()))

        if row["delta_prev"]:
            d = row["delta_prev"]
            improved = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v > 0]
            declined  = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v < 0]
            stable    = [k for k, v in d.items() if k in _TASK_LABELS and v is not None and v == 0]
            lines.append("delta_vs_prev: " + ", ".join(f"{k}={fmt_delta(v)}" for k, v in d.items()))
            lines.append(f"improved_tasks={improved}  declined_tasks={declined}  stable_tasks={stable}")
        else:
            lines.append("delta_vs_prev: baseline iteration (no previous to compare against)")

        if row["best_iteration_so_far"] == row["iteration"]:
            lines.append("status: ★ NEW BEST average so far")
        else:
            lines.append(f"status: below best-so-far iteration {row['best_iteration_so_far']}")
        lines.append("")

    path.write_text("\n".join(lines))
    logger.info(f"Comparative summary written to: {path}")


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------

def _make_leaderboard_md(row: dict) -> str:
    """Render a single iteration's results as a Markdown table for TensorBoard text cards.

    TensorBoard's text plugin renders basic Markdown, so a table gives evaluators
    a quick at-a-glance scorecard directly inside the TensorBoard UI.
    """
    m = row["metrics"]
    dp = row["delta_prev"]
    lines = [
        f"## {row['iteration']} — Benchmark Scores",
        "",
        "| Task | Score | ΔPrev |",
        "|------|------:|------:|",
    ]
    for _, label, _ in TASKS:
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


def log_tensorboard(rows: list[dict], tb_dir: Path, args: argparse.Namespace) -> None:
    """Write all evaluation metrics to TensorBoard.

    Scalar tags written per iteration (step = iteration number):
      eval/average                  — macro-average score across all tasks (%)
      eval/best_so_far_average      — running maximum average seen so far (%)
      eval/task_count               — number of tasks that completed successfully
      eval/tasks/<task_label>       — individual task scores (%)  [as scalars]
      compare_vs_prev/<task_label>  — score delta vs previous iteration (pp)
      compare_vs_prev/average       — average delta vs previous (pp)
      compare_vs_prev/improved_task_count  — number of tasks that improved
      compare_vs_prev/declined_task_count  — number of tasks that declined
      compare_vs_prev/improvement_rate     — improved_count / total_tasks (0–1)

    Text cards written per iteration:
      eval/scorecard                — Markdown table with per-task scores and deltas
      eval/best_iteration           — note when a new best average is reached

    Metadata text card (written once at step 0):
      eval/run_config               — shot counts, example limit, device, checkpoint dir
    """
    logger.info(f"Writing TensorBoard metrics to: {tb_dir}")
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Register custom scalar layout so TensorBoard groups related metrics under
    # named sections in the "Custom Scalars" dashboard tab
    writer.add_custom_scalars({
        "Evaluation": {
            "Average Score (%)": ["Multiline", [_TB_TAG_AVG, _TB_TAG_BEST_AVG]],
            "Per-Task Scores":   ["Multiline", [f"{_TB_TAG_TASKS}/{t[1].lower()}" for t in TASKS]],
        },
        "Delta vs Previous": {
            "Average Delta":       ["Multiline", [f"{_TB_TAG_DELTA_PREFIX}average"]],
            "Per-Task Deltas":     ["Multiline", [f"{_TB_TAG_DELTA_PREFIX}{t[1].lower()}" for t in TASKS]],
            "Task Change Counts":  ["Multiline", [_TB_TAG_IMPROVED_COUNT, _TB_TAG_DECLINED_COUNT]],
            "Improvement Rate":    ["Multiline", [_TB_TAG_IMPROVEMENT_RATE]],
        },
    })

    # ── Run-level metadata card (written once so it's easy to find in the TEXT tab) ──
    shot_summary = ", ".join(f"{k}={v}" for k, v in sorted(args.n_shots_resolved.items()))
    config_md = (
        "## Evaluation Run Configuration\n\n"
        f"| Setting | Value |\n"
        f"|---------|-------|\n"
        f"| Checkpoints dir | `{args.checkpoints_dir}` |\n"
        f"| Output dir | `{args.output_dir}` |\n"
        f"| Device | `{args.device}` |\n"
        f"| Example limit | `{args.limit if args.limit else 'full dataset'}` |\n"
        f"| Shot counts | `{shot_summary}` |\n"
        f"| Iterations evaluated | `{len(rows)}` |\n"
    )
    writer.add_text(_TB_TAG_RUN_CONFIG, config_md, global_step=0)

    for row in rows:
        # Use the iteration number as the TensorBoard x-axis step so that
        # iter_0, iter_1, … line up with a natural integer axis
        step = iter_num(row["iteration"])
        metrics = row["metrics"]

        # ── Core scalar metrics ──────────────────────────────────────────────
        avg = metrics.get("Average")
        if avg is not None:
            writer.add_scalar(_TB_TAG_AVG, avg, step)

        # Log each task as both an individual scalar and as part of the grouped scalars
        task_scores: dict[str, float] = {
            t[1].lower(): metrics[t[1]]
            for t in TASKS if metrics.get(t[1]) is not None
        }
        if task_scores:
            # add_scalars writes one event with multiple series — shows in the same chart
            writer.add_scalars(_TB_TAG_TASKS, task_scores, step)
            # Also write individually so each task has its own chart under eval/tasks/<name>
            for name, score in task_scores.items():
                writer.add_scalar(f"{_TB_TAG_TASKS}/{name}", score, step)

        writer.add_scalar(_TB_TAG_TASK_COUNT, len(task_scores), step)

        best_avg = row.get("best_so_far_avg")
        if best_avg is not None:
            writer.add_scalar(_TB_TAG_BEST_AVG, best_avg, step)

        # ── Delta vs previous iteration ──────────────────────────────────────
        if row["delta_prev"]:
            d = row["delta_prev"]
            for k, v in d.items():
                if v is not None:
                    writer.add_scalar(f"{_TB_TAG_DELTA_PREFIX}{k.lower()}", v, step)

            # Count tasks that improved, declined, or stayed the same this iteration
            improved_count = sum(
                1 for k, v in d.items() if k in _TASK_LABELS and v is not None and v > 0
            )
            declined_count = sum(
                1 for k, v in d.items() if k in _TASK_LABELS and v is not None and v < 0
            )
            total_valid = sum(
                1 for k, v in d.items() if k in _TASK_LABELS and v is not None
            )
            writer.add_scalar(_TB_TAG_IMPROVED_COUNT, improved_count, step)
            writer.add_scalar(_TB_TAG_DECLINED_COUNT, declined_count, step)
            # Improvement rate: fraction of tasks that got better this iteration
            if total_valid > 0:
                writer.add_scalar(_TB_TAG_IMPROVEMENT_RATE, improved_count / total_valid, step)

        # ── Text cards ───────────────────────────────────────────────────────
        # Scorecard table — visible in the TEXT tab of TensorBoard
        writer.add_text(_TB_TAG_SCORECARD, _make_leaderboard_md(row), step)

        # New-best annotation — makes it easy to scan the TEXT tab for breakthroughs
        if row["best_iteration_so_far"] == row["iteration"] and avg is not None:
            writer.add_text(
                _TB_TAG_BEST_ITER,
                f"**{row['iteration']}** achieved new best average: **{fmt(avg)}%**",
                step,
            )

    writer.flush()
    writer.close()
    logger.info(f"TensorBoard logging complete: {len(rows)} iteration(s) written to {tb_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate successive SPIN checkpoints on standard benchmarks, compare "
            "iteration-over-iteration, and log metrics to TensorBoard."
        )
    )
    ap.add_argument(
        "--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR,
        help="Root directory containing iter_* sub-directories.",
    )
    ap.add_argument(
        "--iters", nargs="*", default=None,
        help="Optional subset of iteration names to evaluate, e.g. iter_0 iter_1.",
    )
    ap.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-iteration JSON files and the comparative summary.",
    )
    ap.add_argument(
        "--tensorboard-dir", default=DEFAULT_TENSORBOARD_DIR,
        help="TensorBoard log directory for evaluation metrics.",
    )
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Max examples per task for smoke testing. Do NOT use for real benchmarks.",
    )
    ap.add_argument(
        "--n-shots", nargs="*", default=None,
        help="Per-task shot count overrides: arc_challenge=10 gsm8k=3 (etc.).",
    )
    args = ap.parse_args()

    # Merge user-provided shot overrides into the standard defaults
    n_shots = dict(DEFAULT_SHOTS)
    if args.n_shots:
        for item in args.n_shots:
            k, _, v = item.partition("=")
            n_shots[k.strip()] = int(v.strip())
    # Attach the resolved shot map to args so log_tensorboard can include it in metadata
    args.n_shots_resolved = n_shots

    logger.info("evaluate.py starting")
    logger.info(f"  Checkpoints dir : {args.checkpoints_dir}")
    logger.info(f"  Output dir      : {args.output_dir}")
    logger.info(f"  TensorBoard dir : {args.tensorboard_dir}")
    logger.info(f"  Device          : {args.device}")
    logger.info(f"  Example limit   : {args.limit if args.limit else 'full dataset'}")
    logger.info(f"  Shot counts     : { {task_id: n_shots[task_id] for task_id, _, _ in TASKS} }")

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir  = Path(args.output_dir)
    tb_dir   = Path(args.tensorboard_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Discover which iteration directories to evaluate
    if args.iters:
        iters = [ckpt_dir / name for name in args.iters]
        logger.info(f"Evaluating specified iterations: {args.iters}")
    else:
        # Auto-discover all iter_* directories and sort by numeric index
        iters = sorted(
            [p for p in ckpt_dir.iterdir() if p.is_dir() and re.match(r"iter_\d+", p.name)],
            key=lambda p: iter_num(p.name),
        )
        logger.info(f"Auto-discovered {len(iters)} iteration(s) in {ckpt_dir}")

    if not iters:
        raise SystemExit(f"No iteration directories found in {ckpt_dir}")

    logger.info(f"Iterations to evaluate: {[p.name for p in iters]}")
    print(f"Found {len(iters)} iteration(s) to evaluate: {[p.name for p in iters]}")
    if args.limit:
        logger.warning(f"SMOKE TEST: limit={args.limit} examples per task — do not use for real benchmarks.")
        print(f"[SMOKE TEST] Limiting to {args.limit} examples per task — do not use for real benchmarks.")

    # State tracked across iterations
    rows: list[dict] = []
    best_avg: float | None  = None      # best average score seen so far across all iterations
    best_iter: str | None   = None      # name of the iteration that achieved best_avg
    prev_metrics: dict | None = None    # metrics from the immediately preceding iteration

    run_start = time.time()

    for iter_path in iters:
        iter_name  = iter_path.name
        model_path = find_model_path(iter_path)
        logger.info(f"========== Evaluating {iter_name} ==========")
        logger.info(f"  Model path: {model_path}")
        print(f"\n{'='*60}", flush=True)
        print(f" {iter_name}  →  {model_path}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            model, tokenizer = load_model_and_tokenizer(str(model_path), args.device)
        except Exception as exc:
            logger.error(f"Failed to load model for {iter_name}: {exc}")
            continue

        metrics, elapsed = run_all_benchmarks(model, tokenizer, args.device, n_shots, args.limit)

        # Free GPU memory immediately — each checkpoint may be several GB
        logger.info(f"Freeing model for {iter_name} from GPU memory.")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Persist the raw scores for this iteration
        json_path = out_dir / f"{iter_name}.parsed.json"
        json_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Scores saved to: {json_path}")

        # Compute deltas only when we have a previous iteration to compare against
        dprev = delta(metrics, prev_metrics) if prev_metrics is not None else None

        avg = metrics["Average"]
        if avg is not None and (best_avg is None or avg > best_avg):
            best_avg  = avg
            best_iter = iter_name
            logger.info(f"New best average: {best_avg:.2f}% at {best_iter}")

        rows.append({
            "iteration":             iter_name,
            "metrics":               metrics,
            "delta_prev":            dprev,
            "best_so_far_avg":       best_avg,
            "best_iteration_so_far": best_iter,
            "elapsed_seconds":       elapsed,
        })
        prev_metrics = metrics

        # ── Per-iteration CLI summary ────────────────────────────────────────
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

    # ── Final summary output ─────────────────────────────────────────────────
    print_results_table(rows)

    total_wall = time.time() - run_start
    logger.info(f"All evaluations complete. Total wall-clock time: {total_wall/60:.1f} min")
    print(f"Total evaluation time: {total_wall/60:.1f} min", flush=True)

    # ── Write outputs ────────────────────────────────────────────────────────
    summary_path = out_dir / "comparative_summary.txt"
    write_summary(rows, summary_path)
    json_summary = out_dir / "comparative_summary.json"
    json_summary.write_text(json.dumps(rows, indent=2))
    logger.info(f"JSON summary written to: {json_summary}")
    log_tensorboard(rows, tb_dir, args)

    logger.info(f"Best iteration overall: {best_iter} ({fmt(best_avg)}%)")
    print(f"\nSaved summary : {summary_path}")
    print(f"TensorBoard   : {tb_dir}")
    print(f"  python -m tensorboard.main --logdir={tb_dir.parent}")
    print(f"\nBest iteration: {best_iter} ({fmt(best_avg)}%)")


if __name__ == "__main__":
    main()
