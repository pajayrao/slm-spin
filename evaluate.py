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
  python evaluate.py --limit 50                          # smoke test
  python evaluate.py --tasks arc_challenge gsm8k         # subset of tasks
  python evaluate.py --skip-tasks hellaswag mmlu         # exclude slow tasks
  python evaluate.py --n-shots arc_challenge=10 gsm8k=3  # override shot counts
"""

from torch.utils.tensorboard import SummaryWriter
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
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
_TRUST_REMOTE_CODE_DATASETS: bool = True

# ---------------------------------------------------------------------------
# Generation (GSM8k)
# ---------------------------------------------------------------------------

GSM8K_MAX_NEW_TOKENS: int = 256

# ---------------------------------------------------------------------------
# Default CLI argument values
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = "./spin_outputs_1"
DEFAULT_CHECKPOINTS_DIR = os.path.join(DEFAULT_BASE_DIR, "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_BASE_DIR, "eval_results")
DEFAULT_TENSORBOARD_DIR = os.path.join(
    DEFAULT_BASE_DIR, "tensorboard/eval_compare")
DEFAULT_DEVICE = "cuda"

# ---------------------------------------------------------------------------
# TensorBoard tag strings
# ---------------------------------------------------------------------------

_TB_TAG_AVG = "eval/average"
_TB_TAG_BEST_AVG = "eval/best_so_far_average"
_TB_TAG_TASK_COUNT = "eval/task_count"
_TB_TAG_TASKS = "eval/tasks"
_TB_TAG_SCORECARD = "eval/scorecard"
_TB_TAG_BEST_ITER = "eval/best_iteration"
_TB_TAG_RUN_CONFIG = "eval/run_config"
_TB_TAG_DELTA_PREFIX = "compare_vs_prev/"
_TB_TAG_IMPROVED_COUNT = "compare_vs_prev/improved_task_count"
_TB_TAG_DECLINED_COUNT = "compare_vs_prev/declined_task_count"
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
DEFAULT_SHOTS:  dict[str, int] = {tid: shots for tid, _, _, shots, _ in TASKS}
DATASET_LIMITS: dict[str, Optional[int]] = {
    tid: lim for tid, _, _, _,     lim in TASKS}

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
        lookup[label.lower()] = task

    resolved: list[tuple] = []
    unknown:  list[str] = []
    seen:     set[str] = set()
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
            logger.info(
                f"  Found model root at candidate '{cand}': {p.resolve()}")
            return p.resolve()
        if cand == "." and any((iter_dir / m).exists() for m in _MODEL_CONFIG_MARKERS):
            logger.info(
                f"  Found model root at iteration directory itself: {iter_dir.resolve()}")
            return iter_dir.resolve()

    logger.info("  No candidate matched — falling back to recursive search.")
    for marker in _MODEL_CONFIG_MARKERS:
        found = list(iter_dir.rglob(marker))
        if found:
            logger.info(
                f"  Recursive search found {marker} at: {found[0].parent.resolve()}")
            return found[0].parent.resolve()

    logger.warning(
        f"  Could not locate a model config under {iter_dir}; will try iter_dir directly.")
    return iter_dir.resolve()


def load_model_and_tokenizer(model_path: str, device: str, attn_implementation: str = "sdpa"):
    """Load a causal LM and its tokenizer from a local checkpoint directory."""
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=_TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  pad_token was None — set to eos_token.")
    tokenizer.padding_side = "left"  # required for correct batched generation

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(
        f"Loading model: dtype={torch_dtype}, device={device}, attn={attn_implementation}")
    kwargs = dict(dtype=torch_dtype, trust_remote_code=_TRUST_REMOTE_CODE)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, **kwargs).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model loaded: {n_params:.0f}M parameters, eval mode.")
    return model, tokenizer


def maybe_compile_model(model, backend: str = "inductor"):
    """Wrap model with torch.compile() for faster inference throughput."""
    if not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile not available (requires PyTorch >= 2.0); skipping.")
        return model
    logger.info(f"Compiling model with backend={backend!r}...")
    try:
        model = torch.compile(model, backend=backend)
        logger.info("  Model compiled successfully.")
    except Exception as exc:
        logger.warning(
            f"  torch.compile failed ({exc}); running in eager mode.")
    return model


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

    for cont in continuations:
        full_ids: list[int] = tokenizer(
            context + cont, add_special_tokens=True)["input_ids"]
        cont_raw: list[int] = tokenizer(
            cont, add_special_tokens=False)["input_ids"]
        cont_len = len(cont_raw)
        if cont_len == 0:
            seq_data.append(([], 0, 0))
            continue
        cont_start = len(full_ids) - cont_len
        if len(full_ids) > MAX_SEQ_LEN:
            full_ids = full_ids[-MAX_SEQ_LEN:]
            cont_start = max(1, len(full_ids) - cont_len)
        seq_data.append((full_ids, cont_start, cont_len))

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

    input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attn_mask_t = torch.tensor(attn_masks,  dtype=torch.long, device=device)
    logits = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits
    log_probs = F.log_softmax(logits, dim=-1)

    scores: list[float] = []
    for i, (full_ids, _, cont_len) in enumerate(seq_data):
        if not full_ids or cont_len == 0:
            scores.append(0.0)
            continue
        cs = adj_starts[i]
        actual_len = min(cont_len, max_len - cs)
        cont_tok_ids = input_ids_t[i, cs:cs + actual_len]
        pred_lp = log_probs[i, cs - 1:cs - 1 + actual_len]
        scores.append(pred_lp[torch.arange(
            actual_len, device=device), cont_tok_ids].sum().item())

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

    flat: list[tuple[list[int], int, int, int, int]] = []
    for ex_idx, (context, continuations) in enumerate(examples):
        for cont_idx, cont in enumerate(continuations):
            full_ids: list[int] = tokenizer(
                context + cont, add_special_tokens=True)["input_ids"]
            cont_len = len(
                tokenizer(cont, add_special_tokens=False)["input_ids"])
            if cont_len == 0:
                continue
            cont_start = len(full_ids) - cont_len
            if len(full_ids) > MAX_SEQ_LEN:
                full_ids = full_ids[-MAX_SEQ_LEN:]
                cont_start = max(1, len(full_ids) - cont_len)
            flat.append((full_ids, cont_start, cont_len, ex_idx, cont_idx))

    for i in range(0, len(flat), rows_per_batch):
        batch = flat[i: i + rows_per_batch]
        max_len = max(len(r[0]) for r in batch)

        padded, masks, adj_cs = [], [], []
        for full_ids, cont_start, *_ in batch:
            pl = max_len - len(full_ids)
            padded.append([pad_id] * pl + full_ids)
            masks.append([0] * pl + [1] * len(full_ids))
            adj_cs.append(cont_start + pl)

        ids_t = torch.tensor(padded, dtype=torch.long, device=device)
        mask_t = torch.tensor(masks,  dtype=torch.long, device=device)
        lp = F.log_softmax(
            model(input_ids=ids_t, attention_mask=mask_t).logits, dim=-1
        )

        for j, (full_ids, _, cont_len, ex_idx, cont_idx) in enumerate(batch):
            cs = adj_cs[j]
            alen = min(cont_len, max_len - cs)
            tok = ids_t[j, cs: cs + alen]
            score = lp[j, cs - 1: cs - 1 +
                       alen][torch.arange(alen, device=device), tok].sum().item()
            output[ex_idx][cont_idx] = score

    return output


def _pick_best(scores: list[float], choices: list[str], normalize: bool) -> int:
    """Select the index of the highest-scoring choice.

    When normalize=True (acc_norm) each score is divided by character length of
    the choice text to prevent bias toward shorter answers.
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
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 32
) -> float:
    """Evaluate ARC-Challenge using length-normalised log-likelihood (acc_norm)."""
    logger.info(
        f"ARC-Challenge: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge",
                      trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  ARC-Challenge: {len(test_examples)} test examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            answer_text = texts[labels.index(ex["answerKey"])]
            few_shot_prefix += f"Question: {ex['question']}\nAnswer: {answer_text}\n\n"

    logger.info("  ARC-Challenge: scoring examples...")
    inputs = [
        (few_shot_prefix + f"Question: {ex['question']}\nAnswer:",
         [f" {t}" for t in ex["choices"]["text"]])
        for ex in test_examples
    ]
    all_scores = score_examples_batched(
        model, tokenizer, inputs, device, batch_size)

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
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 32
) -> float:
    """Evaluate TruthfulQA MC2: softmax probability mass on correct answers."""
    logger.info(f"TruthfulQA MC2: loading dataset (zero-shot, limit={limit})")
    if n_shot != 0:
        logger.warning(
            f"TruthfulQA has no few-shot train split; n_shot={n_shot} ignored.")
    ds = load_dataset("truthful_qa", "multiple_choice",
                      trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    examples = list(ds["validation"])
    if limit:
        examples = examples[:limit]
    logger.info(f"  TruthfulQA MC2: {len(examples)} validation examples.")

    inputs = [
        (f"Q: {ex['question']}\nA:", [
         f" {c}" for c in ex["mc2_targets"]["choices"]])
        for ex in examples
    ]
    all_scores = score_examples_batched(
        model, tokenizer, inputs, device, batch_size)

    mc2_scores: list[float] = []
    for ex_scores, ex in zip(all_scores, examples):
        labels = ex["mc2_targets"]["labels"]
        log_lls = torch.tensor(ex_scores, dtype=torch.float64)
        probs = torch.softmax(log_lls, dim=0)
        mc2_scores.append(
            float(sum(probs[i] for i, lbl in enumerate(labels) if lbl == 1)))
    return float(sum(mc2_scores) / len(mc2_scores))


# ---------------------------------------------------------------------------
# Winogrande  (metric: acc)
# ---------------------------------------------------------------------------

def eval_winogrande(
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 32
) -> float:
    """Evaluate Winogrande commonsense pronoun resolution (acc)."""
    logger.info(
        f"Winogrande: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("winogrande", "winogrande_xl",
                      trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["validation"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  Winogrande: {len(test_examples)} validation examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            answer_option = ex["option1"] if ex["answer"] == "1" else ex["option2"]
            few_shot_prefix += ex["sentence"].replace(
                "_", answer_option) + "\n\n"

    inputs = []
    for ex in test_examples:
        blank_idx = ex["sentence"].index("_")
        context = few_shot_prefix + ex["sentence"][:blank_idx]
        rest = ex["sentence"][blank_idx + 1:]
        inputs.append((context, [ex["option1"] + rest, ex["option2"] + rest]))

    all_scores = score_examples_batched(
        model, tokenizer, inputs, device, batch_size)

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
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 8
) -> float:
    """Evaluate GSM8k grade-school math via batched greedy generation (acc)."""
    logger.info(
        f"GSM8k: loading dataset (n_shot={n_shot}, limit={limit}, max_new_tokens={GSM8K_MAX_NEW_TOKENS})")
    ds = load_dataset(
        "gsm8k", "main", trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(f"  GSM8k: {len(test_examples)} test examples.")

    few_shot_prefix = "".join(
        f"Question: {q}\nAnswer: {a}\n\n"
        for q, a in _GSM8K_FEW_SHOT[:n_shot]
    )

    max_prompt_len = MAX_SEQ_LEN - GSM8K_MAX_NEW_TOKENS
    correct = 0
    for i in tqdm(range(0, len(test_examples), batch_size), desc="GSM8k", leave=False, unit="batch"):
        batch = test_examples[i: i + batch_size]
        prompts = [few_shot_prefix +
                   f"Question: {ex['question']}\nAnswer:" for ex in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for j, ex in enumerate(batch):
            generated = tokenizer.decode(
                out[j, prompt_len:], skip_special_tokens=True)
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
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 32
) -> float:
    """Evaluate HellaSwag commonsense sentence completion (acc_norm)."""
    logger.info(f"HellaSwag: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("Rowan/hellaswag",
                      trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    val_examples = list(ds["validation"])
    if limit:
        val_examples = val_examples[:limit]
    logger.info(f"  HellaSwag: {len(val_examples)} validation examples.")

    few_shot_prefix = ""
    if n_shot > 0:
        for ex in list(ds["train"])[:n_shot]:
            ctx = _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"])
            best_ending = _clean_hellaswag(ex["endings"][int(ex["label"])])
            few_shot_prefix += f"{ctx} {best_ending}\n\n"

    inputs = [
        (
            few_shot_prefix +
            _clean_hellaswag(ex["activity_label"] + ": " + ex["ctx"]),
            [" " + _clean_hellaswag(e) for e in ex["endings"]],
        )
        for ex in val_examples
    ]
    all_scores = score_examples_batched(
        model, tokenizer, inputs, device, batch_size)

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
    model, tokenizer, device: str, n_shot: int, limit: Optional[int], batch_size: int = 32
) -> float:
    """Evaluate MMLU across 57 subjects using single-letter continuation scoring (acc)."""
    logger.info(f"MMLU: loading dataset (n_shot={n_shot}, limit={limit})")
    ds = load_dataset("cais/mmlu", "all",
                      trust_remote_code=_TRUST_REMOTE_CODE_DATASETS)
    test_examples = list(ds["test"])
    if limit:
        test_examples = test_examples[:limit]
    logger.info(
        f"  MMLU: {len(test_examples)} test examples across all subjects.")

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

    all_scores = score_examples_batched(
        model, tokenizer, inputs, device, batch_size)

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
    batch_size: int = 32,
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
    logger.info(
        f"Starting benchmark suite: {len(active_tasks)} tasks, device={device}, limit={limit}")

    for task_id, label, _, _, task_limit in active_tasks:
        n_shot = n_shots.get(task_id, 0)
        effective_limit = task_limit if task_limit is not None else limit
        n_examples = f"limit={effective_limit}" if effective_limit else "full"
        logger.info(
            f"--- Task: {label} ({task_id}) | {n_shot}-shot | {n_examples} ---")
        print(
            f"  [{label}] {task_id} | {n_shot}-shot | {n_examples} ...", flush=True)
        t0 = time.time()
        try:
            score = _EVAL_FNS[task_id](
                model, tokenizer, device,
                n_shot=n_shot,
                limit=effective_limit,
                batch_size=batch_size,
            )
            results[label] = round(score * 100.0, 4)
            elapsed[label] = round(time.time() - t0, 1)
            logger.info(
                f"  {label} complete: {results[label]:.2f}%  ({elapsed[label]:.0f}s)")
            print(
                f"    {label}: {results[label]:.2f}%  ({elapsed[label]:.0f}s)", flush=True)
        except Exception as exc:
            elapsed[label] = round(time.time() - t0, 1)
            logger.warning(
                f"Task {task_id} failed after {elapsed[label]:.0f}s: {exc}")
            results[label] = None

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
    sign = "+" if v > 0 else ""
    return f"{arrow}{sign}{v:.2f}"


def iter_num(name: str) -> int:
    m = re.search(r"iter_(\d+)", name)
    return int(m.group(1)) if m else _ITER_NUM_SENTINEL


# ---------------------------------------------------------------------------
# CLI results table
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    """Print a formatted ASCII table of all iteration results to stdout."""
    col_labels = ["Iteration", "Arc", "TruthfulQA", "Winogrande",
                  "GSM8k", "HellaSwag", "MMLU", "Avg%", "ΔAvg", "Status"]

    data: list[list[str]] = []
    for row in rows:
        m = row["metrics"]
        dp = row["delta_prev"]
        is_best = row["best_iteration_so_far"] == row["iteration"]
        delta_avg = fmt_delta(dp.get("Average") if dp else None)
        data.append([
            row["iteration"],
            fmt(m.get("Arc")),        fmt(m.get("TruthfulQA")),
            fmt(m.get("Winogrande")), fmt(m.get("GSM8k")),
            fmt(m.get("HellaSwag")),  fmt(m.get("MMLU")),
            fmt(m.get("Average")),    delta_avg,
            "★ BEST" if is_best else "",
        ])

    widths = [max(len(col_labels[i]), max((len(r[i])
                  for r in data), default=0)) for i in range(len(col_labels))]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_row = "| " + \
        " | ".join(col_labels[i].ljust(widths[i])
                   for i in range(len(col_labels))) + " |"

    print("\n" + sep)
    print(header_row)
    print(sep)
    for row_cells in data:
        print("| " + " | ".join(row_cells[i].ljust(widths[i])
              for i in range(len(col_labels))) + " |")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(rows: list[dict], path: Path) -> None:
    """Write a human-readable comparative summary to a TSV + narrative text file."""
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
        lines.append(
            "scores: " + ", ".join(f"{k}={fmt(v)}" for k, v in m.items()))

        if row["delta_prev"]:
            d = row["delta_prev"]
            improved = [k for k, v in d.items(
            ) if k in _TASK_LABELS and v is not None and v > 0]
            declined = [k for k, v in d.items(
            ) if k in _TASK_LABELS and v is not None and v < 0]
            stable = [k for k, v in d.items(
            ) if k in _TASK_LABELS and v is not None and v == 0]
            lines.append("delta_vs_prev: " +
                         ", ".join(f"{k}={fmt_delta(v)}" for k, v in d.items()))
            lines.append(
                f"improved_tasks={improved}  declined_tasks={declined}  stable_tasks={stable}")
        else:
            lines.append(
                "delta_vs_prev: baseline iteration (no previous to compare against)")

        if row["best_iteration_so_far"] == row["iteration"]:
            lines.append("status: ★ NEW BEST average so far")
        else:
            lines.append(
                f"status: below best-so-far iteration {row['best_iteration_so_far']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparative summary written to: {path}")


# ---------------------------------------------------------------------------
# TensorBoard logging
# ---------------------------------------------------------------------------

def _make_leaderboard_md(row: dict) -> str:
    """Render a single iteration's results as a Markdown table for TensorBoard."""
    m = row["metrics"]
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
    lines.append(
        f"| **Average** | **{fmt(m.get('Average'))}%** | **{fmt_delta(dp.get('Average') if dp else None)}** |")
    lines.append("")
    if row["best_iteration_so_far"] == row["iteration"]:
        lines.append(f"**★ New best average: {fmt(m.get('Average'))}%**")
    else:
        lines.append(
            f"Best so far: {row['best_iteration_so_far']} ({fmt(row.get('best_so_far_avg'))}%)")
    return "\n".join(lines)


def _tb_setup(
    tb_dir: Path,
    args: argparse.Namespace,
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

    active_ids = [t[0] for t in active_tasks]
    shot_summary = ", ".join(
        f"{k}={v}" for k, v in sorted(args.n_shots_resolved.items()) if k in active_ids
    )
    task_summary = ", ".join(f"{t[0]} ({t[1]})" for t in active_tasks)
    config_md = (
        "## Evaluation Run Configuration\n\n"
        f"| Setting | Value |\n"
        f"|---------|-------|\n"
        f"| Checkpoints dir | `{args.checkpoints_dir}` |\n"
        f"| Output dir | `{args.output_dir}` |\n"
        f"| Device | `{args.device}` |\n"
        f"| Example limit | `{args.limit if args.limit else 'full dataset'}` |\n"
        f"| Active tasks | `{task_summary}` |\n"
        f"| Shot counts | `{shot_summary}` |\n"
        f"| Iterations planned | `{n_iters}` |\n"
    )
    writer.add_text(_TB_TAG_RUN_CONFIG, config_md, global_step=0)
    return writer


def _tb_write_row(writer: "SummaryWriter", row: dict) -> None:
    """Write one iteration's metrics to an already-open SummaryWriter and flush."""
    step = iter_num(row["iteration"])
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
                writer.add_scalar(
                    f"{_TB_TAG_DELTA_PREFIX}{k.lower()}", v, step)

        improved_count = sum(1 for k, v in d.items()
                             if k in _TASK_LABELS and v is not None and v > 0)
        declined_count = sum(1 for k, v in d.items()
                             if k in _TASK_LABELS and v is not None and v < 0)
        total_valid = sum(1 for k, v in d.items()
                          if k in _TASK_LABELS and v is not None)
        writer.add_scalar(_TB_TAG_IMPROVED_COUNT, improved_count, step)
        writer.add_scalar(_TB_TAG_DECLINED_COUNT, declined_count, step)
        if total_valid > 0:
            writer.add_scalar(_TB_TAG_IMPROVEMENT_RATE,
                              improved_count / total_valid, step)

    writer.add_text(_TB_TAG_SCORECARD, _make_leaderboard_md(row), step)

    if row["best_iteration_so_far"] == row["iteration"] and avg is not None:
        writer.add_text(
            _TB_TAG_BEST_ITER,
            f"**{row['iteration']}** achieved new best average: **{fmt(avg)}%**",
            step,
        )

    writer.flush()
    logger.info(
        f"TensorBoard: flushed metrics for {row['iteration']} (step={step})")


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
        "--attn-impl", default="sdpa",
        help="Attention implementation: sdpa (default), flash_attention_2, or empty string for eager.",
    )
    ap.add_argument(
        "--compile", action="store_true", default=True,
        help="Apply torch.compile() to each checkpoint model before evaluation (requires PyTorch >= 2.0).",
    )
    ap.add_argument(
        "--compile-backend", default="inductor",
        help="torch.compile backend (default: inductor). Use aot_eager if triton is unavailable.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Max examples per task for smoke testing. Do NOT use for real benchmarks.",
    )
    ap.add_argument(
        "--n-shots", nargs="*", default=None,
        help="Per-task shot count overrides: arc_challenge=10 gsm8k=3 (etc.).",
    )
    ap.add_argument(
        "--eval-batch-size", type=int, default=8,
        help="Number of (context, continuation) rows per GPU forward pass (default: 8). "
             "Increase for shorter sequences or larger GPUs; decrease if OOM.",
    )
    ap.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Re-evaluate iterations even if a cached JSON result already exists.",
    )

    valid_ids = [t[0] for t in TASKS]
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
    ap.add_argument("--tasks",      nargs="+", default=None,
                    metavar="TASK", help=task_help)
    ap.add_argument("--skip-tasks", nargs="+", default=None,
                    metavar="TASK", help=skip_help)

    args = ap.parse_args()

    if args.tasks and args.skip_tasks:
        raise SystemExit("--tasks and --skip-tasks are mutually exclusive.")

    # Build active_tasks by whitelist or blacklist
    if args.tasks:
        active_tasks = resolve_task_filter(args.tasks)
    elif args.skip_tasks:
        skip_ids = {t[0] for t in resolve_task_filter(args.skip_tasks)}
        active_tasks = [t for t in TASKS if t[0] not in skip_ids]
    else:
        active_tasks = list(TASKS)

    if not active_tasks:
        raise SystemExit(
            "No tasks selected — check --tasks / --skip-tasks arguments.")

    # Merge user-provided shot overrides into the per-task defaults from TASKS
    n_shots = dict(DEFAULT_SHOTS)
    if args.n_shots:
        for item in args.n_shots:
            k, _, v = item.partition("=")
            n_shots[k.strip()] = int(v.strip())
    args.n_shots_resolved = n_shots

    logger.info("evaluate.py starting")
    logger.info(f"  Checkpoints dir : {args.checkpoints_dir}")
    logger.info(f"  Output dir      : {args.output_dir}")
    logger.info(f"  TensorBoard dir : {args.tensorboard_dir}")
    logger.info(f"  Device          : {args.device}")
    logger.info(
        f"  Example limit   : {args.limit if args.limit else 'full dataset'}")
    logger.info(f"  Active tasks    : {[t[0] for t in active_tasks]}")
    logger.info(
        f"  Shot counts     : { {tid: n_shots[tid] for tid, *_ in active_tasks} }")

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    tb_dir = Path(args.tensorboard_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Discover which iteration directories to evaluate
    if args.iters:
        iters = [ckpt_dir / name for name in args.iters]
        logger.info(f"Evaluating specified iterations: {args.iters}")
    else:
        iters = sorted(
            [p for p in ckpt_dir.iterdir() if p.is_dir()
             and re.match(r"iter_\d+", p.name)],
            key=lambda p: iter_num(p.name),
        )
        logger.info(f"Auto-discovered {len(iters)} iteration(s) in {ckpt_dir}")

    if not iters:
        raise SystemExit(f"No iteration directories found in {ckpt_dir}")

    logger.info(f"Iterations to evaluate: {[p.name for p in iters]}")
    print(
        f"Found {len(iters)} iteration(s) to evaluate: {[p.name for p in iters]}")
    if args.limit:
        logger.warning(
            f"SMOKE TEST: limit={args.limit} examples per task — do not use for real benchmarks.")
        print(f"[SMOKE TEST] Limiting to {args.limit} examples per task.")

    # State tracked across iterations
    rows:         list[dict] = []
    best_avg:     float | None = None
    best_iter:    str | None = None
    prev_metrics: dict | None = None

    run_start = time.time()
    tb_writer = _tb_setup(tb_dir, args, len(iters), active_tasks)

    for iter_path in iters:
        iter_name = iter_path.name
        json_path = out_dir / f"{iter_name}.parsed.json"
        print(f"\n{'='*60}", flush=True)

        # Use cached result if available and --no-cache not set
        if json_path.exists() and not args.no_cache:
            logger.info(f"========== Loading cached {iter_name} ==========")
            print(f" {iter_name}  →  [cached] {json_path}", flush=True)
            print(f"{'='*60}", flush=True)
            metrics = json.loads(json_path.read_text())
            elapsed: dict[str, float] = {}
        else:
            model_path = find_model_path(iter_path)
            logger.info(f"========== Evaluating {iter_name} ==========")
            logger.info(f"  Model path: {model_path}")
            print(f" {iter_name}  →  {model_path}", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                model, tokenizer = load_model_and_tokenizer(
                    str(model_path), args.device, attn_implementation=args.attn_impl or None,
                )
                if args.compile:
                    model = maybe_compile_model(
                        model, backend=args.compile_backend)
            except Exception as exc:
                logger.error(f"Failed to load model for {iter_name}: {exc}")
                continue

            metrics, elapsed = run_all_benchmarks(
                model, tokenizer, args.device, n_shots, args.limit,
                batch_size=args.eval_batch_size,
                active_tasks=active_tasks,
            )

            logger.info(f"Freeing model for {iter_name} from GPU memory.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            json_path.write_text(json.dumps(metrics, indent=2))
            logger.info(f"Scores saved to: {json_path}")

        # Compute deltas only when we have a previous iteration to compare against
        dprev = delta(
            metrics, prev_metrics) if prev_metrics is not None else None

        avg = metrics["Average"]
        if avg is not None and (best_avg is None or avg > best_avg):
            best_avg = avg
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

        dprev_str = fmt_delta(dprev.get("Average") if dprev else None)
        is_best_str = " ★ NEW BEST" if best_iter == iter_name else ""
        total_time = sum(elapsed.values())
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
            improved = [k for k, v in dprev.items(
            ) if k in _TASK_LABELS and v is not None and v > 0]
            declined = [k for k, v in dprev.items(
            ) if k in _TASK_LABELS and v is not None and v < 0]
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
    logger.info(
        f"All evaluations complete. Total wall-clock time: {total_wall/60:.1f} min")
    print(f"Total evaluation time: {total_wall/60:.1f} min", flush=True)

    summary_path = out_dir / "comparative_summary.txt"
    write_summary(rows, summary_path)
    json_summary = out_dir / "comparative_summary.json"
    json_summary.write_text(json.dumps(rows, indent=2))
    logger.info(f"JSON summary written to: {json_summary}")
    tb_writer.close()

    logger.info(f"Best iteration overall: {best_iter} ({fmt(best_avg)}%)")
    print(f"\nSaved summary : {summary_path}")
    print(f"TensorBoard   : {tb_dir}")
    print(f"  python -m tensorboard.main --logdir={tb_dir.parent}")
    print(f"\nBest iteration: {best_iter} ({fmt(best_avg)}%)")


if __name__ == "__main__":
    main()
