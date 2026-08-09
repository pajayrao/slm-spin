import os
import gc
import json
import argparse
import hashlib
import logging
import glob
import traceback
from typing import List, Dict, Any
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import psutil


import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from spin_config import *

logging_kwargs = {
    "format": '%(asctime)s %(levelname)-8s %(message)s',
    "level": logging.INFO,
    "datefmt": '%Y-%m-%d %H:%M:%S',
}

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


def find_start_iteration(cfg) -> int:
    """Return the first iteration index that has not yet completed.

    Primary signal: .done sentinel, written explicitly after all saves finish.
    Fallback: tokenizer_config.json written by tokenizer.save_pretrained(), which
    runs after trainer.save_model() — supports runs completed before .done was added.
    If neither file exists the iteration is treated as incomplete even if the dir
    was created by ensure_dir() before training started.

    Example:
        Input:  cfg.num_iterations=3, cfg.checkpoints_dir="output/checkpoints"
                Directory state:
                  output/checkpoints/iter_0/.done  ← exists
                  output/checkpoints/iter_1/       ← exists but no .done or tokenizer_config.json
                  output/checkpoints/iter_2/       ← does not exist
        Output: 1  (iteration 0 is done; iteration 1 must be re-run)

        Input:  cfg.num_iterations=3, no checkpoints present at all
        Output: 0  (fresh run)

        Input:  cfg.num_iterations=3, iter_0/.done and iter_1/.done both exist
        Output: 2  (resume at the last incomplete iteration)
    """
    logger.info(
        "find_start_iteration() — scanning checkpoint dirs for the last completed iteration...")
    for i in range(cfg.num_iterations - 1, -1, -1):
        iter_dir = os.path.join(cfg.checkpoints_dir, f"iter_{i}")
        done_sentinel = os.path.join(iter_dir, ".done")
        tok_cfg = os.path.join(iter_dir, "tokenizer_config.json")
        if os.path.exists(done_sentinel):
            logger.info(
                f"  iter_{i}: .done sentinel found → iterations 0–{i} complete, resuming at {i + 1}.")
            return i + 1
        if os.path.exists(tok_cfg):
            logger.info(
                f"  iter_{i}: tokenizer_config.json found (legacy signal) → resuming at {i + 1}.")
            return i + 1
    logger.info(
        "  No completed iterations found — starting from iteration 0 (fresh run).")
    return 0


def pre_start_cleanup():
    """Remove stale HuggingFace lock files left behind by killed processes.

    HF datasets/hub write .lock files to coordinate concurrent downloads.
    If a previous run was killed, those locks are never released and subsequent
    runs hang forever waiting to acquire them.  Deleting them at process start
    is safe because this process is the only one using this cache directory.

    Example:
        Input:  ~/.cache/huggingface/datasets/some_dataset/data.lock  ← stale file on disk
        Output: file is deleted; function returns None and logs
                "Removed stale lock: ~/.cache/huggingface/datasets/some_dataset/data.lock"

        Input:  no .lock files present anywhere under ~/.cache/huggingface
        Output: None (no-op, nothing logged)
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    for lock_file in glob.glob(os.path.join(hf_cache, "**", "*.lock"), recursive=True):
        try:
            os.remove(lock_file)
            logger.info(f"Removed stale lock: {lock_file}")
        except OSError:
            pass

# NOTE: pre_start_cleanup() is deliberately NOT called at import time. Deleting
# every HF cache lock is only safe when this process is the sole user of the
# cache; an import-time call fired from every module that imports utils (main,
# evaluate, sft_warmup, …) including inside concurrently running processes.
# main.py and evaluate.py call it explicitly at process start instead.


def ensure_dir(path: str):
    """Create a directory (and all parents) if it does not already exist.

    Example:
        Input:  path="output/checkpoints/iter_0"  (no parents exist yet)
        Output: directories "output/", "output/checkpoints/", and
                "output/checkpoints/iter_0/" are all created; returns None

        Input:  path="output/checkpoints/iter_0"  (directory already exists)
        Output: None (no-op, no error raised)
    """
    os.makedirs(path, exist_ok=True)


def log_memory(tag: str):
    """Log current CPU RSS and GPU allocated/reserved memory to the Python logger.

    Args:
        tag: A short label (e.g. "before_trainer_init") printed alongside the numbers
             so spikes can be correlated with specific code events in the log.

    Example:
        Input:  tag="before_trainer_init"
                CPU RSS is 4200 MB, GPU allocated is 8192 MB, GPU reserved is 10240 MB
        Output: logs "[MEM before_trainer_init] CPU RSS 4200 MB | GPU alloc 8192 MB | GPU reserved 10240 MB"
                returns None

        Input:  tag="after_load_iter0"  (no CUDA device available)
                CPU RSS is 2048 MB
        Output: logs "[MEM after_load_iter0] CPU RSS 2048 MB"
                returns None
    """
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
        logger.info(
            f"[MEM {tag}] CPU RSS {rss_mb:.0f} MB | GPU alloc {alloc_mb:.0f} MB | GPU reserved {reserved_mb:.0f} MB")
    else:
        logger.info(f"[MEM {tag}] CPU RSS {rss_mb:.0f} MB")


def str2dtype(name: str):
    """Convert a dtype name string to the corresponding torch.dtype.

    Accepted values: "float16", "bfloat16", "float32" (case-insensitive).
    bfloat16 is preferred over float16 on CUDA because it preserves the same
    dynamic range as float32 while halving memory, avoiding the overflow/underflow
    issues that float16 can introduce during LLM training.

    Example:
        Input:  name="bfloat16"
        Output: torch.bfloat16

        Input:  name="FLOAT32"
        Output: torch.float32

        Input:  name="float16"
        Output: torch.float16

        Input:  name="int8"
        Output: raises ValueError("Unsupported torch_dtype: int8")
    """
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def parse_args() -> SPINConfig:
    """Parse CLI arguments and return a populated SPINConfig dataclass.

    Dynamically generates one argparse flag per SPINConfig field so the config
    is the single source of truth — adding a field to SPINConfig automatically
    exposes it as a CLI flag without touching this function.  Boolean fields
    are accepted as strings ("true"/"false"/"1"/"0") because argparse cannot
    natively handle bool defaults without ambiguity.

    Example:
        Input:  sys.argv = ["main.py",
                            "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",
                            "--data_path", "data/train.jsonl",
                            "--num_iterations", "3",
                            "--use_lora", "true",
                            "--learning_rate", "5e-5"]
        Output: SPINConfig(
                    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
                    data_path="data/train.jsonl",
                    num_iterations=3,
                    use_lora=True,
                    learning_rate=5e-05,
                    ... (all other fields set to their defaults)
                )

        Input:  sys.argv = ["main.py"]  (no flags; all defaults)
        Output: SPINConfig()  (fully populated with default values from the dataclass)
    """
    parser = argparse.ArgumentParser(description="SPIN training")

    for field_name, field_def in SPINConfig.__dataclass_fields__.items():
        default = field_def.default
        if isinstance(default, bool):
            parser.add_argument(f"--{field_name}",
                                type=str, default=str(default))
        else:
            parser.add_argument(
                f"--{field_name}",
                type=type(default) if default is not None else str,
                default=default,
            )

    args = parser.parse_args()
    kwargs = {}
    for k, v in vars(args).items():
        default = SPINConfig.__dataclass_fields__[k].default
        if isinstance(default, bool):
            kwargs[k] = str(v).lower() in ("1", "true", "yes", "y")
        else:
            kwargs[k] = v
    return SPINConfig(**kwargs)


# -----------------------------
# Formatting helpers
# -----------------------------

def maybe_apply_chat_template(tokenizer, user_prompt: str, cfg: SPINConfig) -> str:
    """Format a raw user prompt according to the configured chat template mode.

    Modes:
      "plain"                — return the prompt unchanged.
      "instruction_response" — wrap with cfg.instruction_prefix / response_prefix.
      "auto"                 — use the tokenizer's built-in chat template when
                               available (covers LLaMA-3, Mistral, Phi-3, etc.),
                               falling back to instruction_response otherwise.

    Example (mode="plain"):
        Input:  user_prompt="What is the capital of France?", cfg.chat_template_mode="plain"
        Output: "What is the capital of France?"

    Example (mode="instruction_response"):
        Input:  user_prompt="What is the capital of France?",
                cfg.chat_template_mode="instruction_response",
                cfg.instruction_prefix="### Instruction:\n",
                cfg.response_prefix="\n### Response:\n"
        Output: "### Instruction:\nWhat is the capital of France?\n### Response:\n"

    Example (mode="auto", tokenizer has a chat template):
        Input:  user_prompt="What is the capital of France?",
                cfg.chat_template_mode="auto",
                tokenizer is a LLaMA-3 tokenizer with a built-in chat template
        Output: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
                 What is the capital of France?<|eot_id|>
                 <|start_header_id|>assistant<|end_header_id|>\n\n"

    Example (mode="auto", tokenizer has NO chat template):
        Input:  user_prompt="What is the capital of France?",
                cfg.chat_template_mode="auto",
                tokenizer.chat_template is None,
                cfg.instruction_prefix="[INST] ", cfg.response_prefix=" [/INST]"
        Output: "[INST] What is the capital of France? [/INST]"
    """
    if cfg.chat_template_mode == "plain":
        logger.debug(
            "maybe_apply_chat_template: mode=plain — prompt returned unchanged.")
        return user_prompt

    if cfg.chat_template_mode == "instruction_response":
        result = f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"
        logger.debug(
            f"maybe_apply_chat_template: mode=instruction_response — "
            f"wrapped with prefix/suffix, output_len={len(result)} chars."
        )
        return result

    if cfg.chat_template_mode == "auto":
        has_template = hasattr(
            tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
        if has_template:
            try:
                # enable_thinking=False: Qwen3 templates default to thinking mode,
                # which makes generations open with a <think>...</think> block that
                # human dataset responses never contain — a trivial surface feature
                # for the SPIN margin to key on. Non-Qwen3 templates ignore the kwarg.
                result = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                logger.debug(
                    f"maybe_apply_chat_template: mode=auto → used tokenizer.apply_chat_template, "
                    f"output_len={len(result)} chars."
                )
                return result
            except Exception as e:
                logger.debug(
                    f"maybe_apply_chat_template: apply_chat_template failed ({e}), "
                    f"falling back to instruction_response format."
                )
                return f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"
        logger.debug(
            "maybe_apply_chat_template: mode=auto — no chat_template on tokenizer, "
            "falling back to instruction_response format."
        )
        return f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"

    raise ValueError(f"Unknown chat_template_mode: {cfg.chat_template_mode}")


def normalize_chat_dataset_record(example, prompt_field="prompt", response_field="response"):
    """Extract a (prompt, response) pair from a dataset record.

    Tries two formats in order:
      1. Flat fields — reads example[prompt_field] and example[response_field] directly.
         Handles datasets that already have separate prompt and response columns.
      2. Messages list — reads the first user/assistant turn from a 'messages' list,
         tolerating common role-name variants (user/human, assistant/model/gpt/bot).

    Returns None for records that yield no valid pair so they can be silently skipped.

    Example (flat fields):
        Input:  example={"prompt": "What is Python?", "response": "Python is a programming language."}
                prompt_field="prompt", response_field="response"
        Output: {"prompt": "What is Python?", "response": "Python is a programming language."}

    Example (messages list, ShareGPT/UltraChat format):
        Input:  example={"messages": [
                    {"role": "user",      "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."}
                ]}
        Output: {"prompt": "What is Python?", "response": "Python is a programming language."}

    Example (messages list with alternate role names):
        Input:  example={"messages": [
                    {"role": "human", "content": "Explain recursion."},
                    {"role": "gpt",   "content": "Recursion is when a function calls itself."}
                ]}
        Output: {"prompt": "Explain recursion.", "response": "Recursion is when a function calls itself."}

    Example (missing response — skipped):
        Input:  example={"prompt": "Hello", "response": ""}
        Output: None  (empty response falls through to messages check; messages absent → None)

    Example (single-turn messages list — skipped):
        Input:  example={"messages": [{"role": "user", "content": "Hi"}]}
        Output: None  (len(messages) < 2)
    """
    # Format 1: flat prompt/response fields (e.g. custom JSONL, Alpaca-style datasets).
    flat_prompt   = example.get(prompt_field, "")
    flat_response = example.get(response_field, "")
    if flat_prompt and flat_response:
        return {"prompt": str(flat_prompt).strip(), "response": str(flat_response).strip()}

    # Format 2: multi-turn messages list (e.g. ShareGPT, UltraChat).
    messages = example.get("messages", [])
    if not messages or len(messages) < 2:
        return None

    user_prompt = None
    assistant_response = None

    for i, turn in enumerate(messages):
        if i >= 2:
            break
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role in ["user", "human"] and user_prompt is None:
            user_prompt = content
        elif role in ["assistant", "model", "gpt", "bot"] and user_prompt is not None and assistant_response is None:
            assistant_response = content
            break

    if not user_prompt or not assistant_response:
        return None

    return {"prompt": user_prompt, "response": assistant_response}


def load_base_dataset_fixed(dataset_name=None, dataset_config_name=None, split="train_sft",
                             data_path=None, limit=None, prompt_field="prompt", response_field="response"):
    """Load and normalize the base training dataset.

    Accepts either a HuggingFace Hub dataset (dataset_name + optional config +
    split) or a local file (JSONL, JSON, Parquet).  Every record is passed
    through normalize_chat_dataset_record; records that don't yield a valid
    user→assistant pair are silently skipped.  The optional limit cap bounds
    startup time when the source dataset is very large.

    Returns an HFDataset with columns {"prompt": str, "response": str}.
    Raises ValueError if no valid pairs are found after filtering.

    Example (local JSONL file):
        Input:  data_path="data/train.jsonl"  (file contains 1200 records, 50 have empty responses)
                limit=1000
        Output: HFDataset with 1000 rows, columns=["prompt", "response"]
                (stops early once 1000 valid pairs collected; logs "50 skipped")

    Example (HuggingFace Hub):
        Input:  dataset_name="HuggingFaceH4/ultrachat_200k",
                dataset_config_name=None,
                split="train_sft",
                limit=5000
        Output: HFDataset with 5000 rows, columns=["prompt", "response"]

    Example (no valid pairs):
        Input:  data_path="empty_dataset.jsonl"  (all records have blank responses)
        Output: raises ValueError("No valid prompt/response pairs found.")
    """
    logger.info(
        "load_base_dataset_fixed() — loading and normalising training dataset...")
    if data_path:
        ext = data_path.rsplit(".", 1)[-1].lower()
        logger.info(f"  Source: local file — {data_path} (format={ext})")
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            ds = load_dataset("json", data_files=data_path, split="train")
        elif data_path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=data_path, split="train")
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
    else:
        logger.info(f"  Source: HuggingFace Hub — {dataset_name} "
                    f"(config={dataset_config_name}, split={split})")
        ds = load_dataset(dataset_name, dataset_config_name, split=split)

    logger.info(f"  Raw dataset loaded: {len(ds)} total records. "
                f"Normalising to (prompt, response) pairs...")

    rows = []
    skipped = 0

    for ex in ds:
        item = normalize_chat_dataset_record(ex, prompt_field=prompt_field, response_field=response_field)
        if item is None:
            skipped += 1
            continue
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            logger.info(f"  Reached limit of {limit} rows — stopping early.")
            break

    logger.info(f"  Normalisation complete: {len(rows)} valid rows kept, {skipped} skipped "
                f"(missing user→assistant turn or empty content).")

    if len(rows) == 0:
        raise ValueError("No valid prompt/response pairs found.")

    result = HFDataset.from_list(rows)
    logger.info(f"load_base_dataset_fixed() complete — HFDataset with {len(result)} rows, "
                f"columns={result.column_names}.")
    return result


# -----------------------------
# Model loading
# -----------------------------

def load_tokenizer(cfg: SPINConfig):
    """Load and configure the tokenizer for SPIN training.

    Sets pad_token to eos_token when none is defined (required by many models
    so padding doesn't trigger unknown-token errors), and forces left-side
    truncation so the response end (which carries the most training signal)
    is always preserved when sequences exceed max_length.

    Example:
        Input:  cfg.model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
                cfg.tokenizer_name_or_path=None,
                cfg.trust_remote_code=False,
                cfg.truncation_side="left"
        Output: AutoTokenizer instance where:
                  tokenizer.vocab_size == 128256
                  tokenizer.pad_token  == tokenizer.eos_token  ("<|eot_id|>")
                  tokenizer.truncation_side == "left"
                  tokenizer.padding_side    == "left"

        Input:  cfg.tokenizer_name_or_path="my_custom_tokenizer/" (local path)
        Output: AutoTokenizer loaded from that local path, same pad/truncation setup applied
    """
    tok_name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    logger.info(f"Loading tokenizer from: {tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  pad_token was None — set to eos_token.")
    tokenizer.truncation_side = cfg.truncation_side
    tokenizer.padding_side = "left"
    logger.info(
        f"  Tokenizer ready: vocab_size={tokenizer.vocab_size}, truncation_side={cfg.truncation_side}")
    return tokenizer


def maybe_compile_model(model, cfg: SPINConfig, label: str = "model"):
    """Wrap model with torch.compile() using the settings from cfg.

    Falls back gracefully when torch.compile is unavailable (PyTorch < 2.0) or
    when compilation fails, so the caller always gets a usable model back.

    Example (compilation succeeds):
        Input:  model=<LlamaForCausalLM>, label="train_model",
                cfg.compile_backend="inductor", cfg.compile_mode="reduce-overhead",
                cfg.compile_dynamic=False, cfg.compile_fullgraph=False
        Output: <OptimizedModule wrapping LlamaForCausalLM>
                (subsequent forward passes are kernel-fused and faster)

    Example (PyTorch < 2.0, torch.compile not available):
        Input:  model=<LlamaForCausalLM>, cfg as above
        Output: original <LlamaForCausalLM> unchanged; logs a warning and returns the uncompiled model

    Example (compilation fails due to unsupported op):
        Input:  model=<CustomModel with unsupported op>, cfg.compile_fullgraph=True
        Output: original model returned unchanged; logs a warning with the exception message
    """
    if not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile not available (requires PyTorch >= 2.0). Skipping.")
        return model
    logger.info(
        f"Compiling {label} with backend={cfg.compile_backend!r}, "
        f"mode={cfg.compile_mode!r}, fullgraph={cfg.compile_fullgraph}"
    )
    try:
        model = torch.compile(
            model,
            dynamic=cfg.compile_dynamic,
            backend=cfg.compile_backend,
            mode=cfg.compile_mode,
            fullgraph=cfg.compile_fullgraph,
        )
        logger.info(f"  {label} compiled successfully.")
    except Exception as e:
        logger.warning(
            f"  torch.compile failed for {label} ({type(e).__name__}: {e}); falling back to eager mode.\n"
            + traceback.format_exc()
        )
    return model


def load_causal_lm(model_path: str, cfg: SPINConfig, trainable: bool = True):
    """Load a causal language model from a local or Hub path.

    When trainable=False the model is returned in eval mode with all gradients
    disabled — this is the π_ref (reference / previous-iteration) model.
    When trainable=True it is returned in train mode ready for make_trainable()
    to apply LoRA or enable full fine-tuning.  Gradient checkpointing is enabled
    here when requested so use_cache is disabled before any weights move to CUDA.

    Example (frozen reference model):
        Input:  model_path="output/checkpoints/iter_0",
                cfg.torch_dtype="bfloat16", cfg.attn_implementation="flash_attention_2",
                cfg.trust_remote_code=False, trainable=False
        Output: LlamaForCausalLM loaded in bfloat16, eval mode,
                all p.requires_grad=False, model.config.use_cache=True,
                ~1B parameters (~2 GB VRAM in bfloat16)

    Example (trainable model with gradient checkpointing):
        Input:  model_path="meta-llama/Llama-3.2-1B-Instruct",
                cfg.torch_dtype="bfloat16", cfg.gradient_checkpointing=True,
                trainable=True
        Output: LlamaForCausalLM in train mode,
                gradient_checkpointing=True, model.config.use_cache=False,
                ready for make_trainable() to attach LoRA adapters
    """
    logger.info(
        f"Loading causal LM from: {model_path} (trainable={trainable}, dtype={cfg.torch_dtype})")
    kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=str2dtype(cfg.torch_dtype),
    )
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
        logger.info(f"  Attention implementation: {cfg.attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if trainable and cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("  Gradient checkpointing enabled; use_cache disabled.")
    else:
        model.config.use_cache = cfg.generation_use_cache

    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        logger.info("  Model frozen (eval mode, no grad).")
        if cfg.compile_ref_model:
            model = maybe_compile_model(model, cfg, label="ref_model")
    else:
        if cfg.log_trainable_parameters:
            _log_trainable_parameters(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model loaded: {n_params:.0f}M parameters.")
    return model


# -----------------------------
# Tokenization
# -----------------------------

def build_prompt_text(prompt: str, tokenizer, cfg: SPINConfig) -> str:
    """Return the formatted prompt string (chat template applied, no response appended).

    Example:
        Input:  prompt="What is recursion?",
                cfg.chat_template_mode="instruction_response",
                cfg.instruction_prefix="### Instruction:\n",
                cfg.response_prefix="\n### Response:\n"
        Output: "### Instruction:\nWhat is recursion?\n### Response:\n"

    Example (LLaMA-3 auto mode):
        Input:  prompt="What is recursion?", cfg.chat_template_mode="auto",
                tokenizer=LlamaTokenizer (has built-in chat template)
        Output: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
                 What is recursion?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    """
    return maybe_apply_chat_template(tokenizer, prompt, cfg)


def build_full_text(prompt: str, response: str, tokenizer, cfg: SPINConfig) -> str:
    """Concatenate the formatted prompt and response into one string for tokenization.

    Appends eos_token when cfg.add_eos_to_response is True and the response
    doesn't already end with it, so the model learns to terminate cleanly.

    Example (add_eos_to_response=True):
        Input:  prompt="What is recursion?",
                response="Recursion is when a function calls itself.",
                cfg.chat_template_mode="plain",
                cfg.add_eos_to_response=True,
                tokenizer.eos_token="</s>"
        Output: "What is recursion?Recursion is when a function calls itself.</s>"

    Example (response already ends with eos):
        Input:  prompt="What is recursion?",
                response="Recursion is when a function calls itself.</s>",
                cfg.add_eos_to_response=True,
                tokenizer.eos_token="</s>"
        Output: "What is recursion?Recursion is when a function calls itself.</s>"
                (eos not appended twice)

    Example (add_eos_to_response=False):
        Input:  prompt="Hi", response="Hello there.", cfg.add_eos_to_response=False
        Output: "HiHello there."
    """
    txt = build_prompt_text(prompt, tokenizer, cfg) + response
    if cfg.add_eos_to_response and tokenizer.eos_token and not txt.endswith(tokenizer.eos_token):
        txt += tokenizer.eos_token
    return txt


def tokenize_prompt_response(tokenizer, prompt: str, response: str, cfg: SPINConfig) -> Dict[str, Any]:
    """Tokenize a prompt+response pair and produce supervised-learning labels.

    The prompt and full text are tokenized together (not separately) to avoid
    boundary artifacts — tokenizers can split subwords differently at the
    boundary when strings are encoded in isolation.  Prompt token positions are
    masked to -100 in `labels` so the loss is computed only over the response.

    Returns a dict with keys: input_ids, attention_mask, labels.

    Example:
        Input:  prompt="What is Python?",
                response="Python is a high-level programming language.",
                cfg.chat_template_mode="plain",
                cfg.max_prompt_length=128, cfg.max_length=256,
                cfg.add_eos_to_response=True, tokenizer.eos_token="</s>"

        Suppose the tokenizer encodes:
          prompt_text → [1, 1724, 338, 5132, 29973]           (5 tokens)
          full_text   → [1, 1724, 338, 5132, 29973, 5132, ...]  (20 tokens total)

        Output: {
            "input_ids":      [1, 1724, 338, 5132, 29973, 5132, 338, 263, 1880, 29899,
                               5563, 8720, 4086, 29889, 2],   # 20 tokens incl. </s>
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1],                 # all 1s (no padding)
            "labels":         [-100, -100, -100, -100, -100,   # 5 prompt tokens masked
                               5132, 338, 263, 1880, 29899,    # response tokens active
                               5563, 8720, 4086, 29889, 2]
        }

    Example (truncation — response is longer than max_length):
        Input:  cfg.max_length=10, full_text tokenizes to 25 tokens
        Output: input_ids truncated to the first 10 tokens from the right
                (cfg.truncation_side="left" → prompt head dropped, response tail kept)
    """
    # Step 1: Build the formatted text strings for prompt alone and prompt+response.
    # Done separately because we need to tokenize both to find where the prompt ends
    # inside the joint sequence (subword tokenizers can split differently at boundaries
    # when strings are encoded in isolation, so we don't just concatenate token lists).
    prompt_text = build_prompt_text(prompt, tokenizer, cfg)
    full_text = build_full_text(prompt, response, tokenizer, cfg)

    # Step 2: Tokenize without truncation to find the exact prompt/response boundary.
    # prompt_ids_full gives us the prompt token count before any truncation.
    # Example: prompt="What is Python?" → prompt_ids_full=[1,1724,338,5132,29973] (5 tokens)
    #          full_text includes response → full_ids_full has 20 tokens total
    prompt_ids_full = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids_full   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]

    # Step 3: Apply left-truncation manually to preserve response tokens over prompt tokens.
    # We truncate from the left (drop prompt head) rather than the right (drop response tail)
    # so the model always sees the full response during training.
    # n_prompt_remaining tracks how many prompt tokens survived after the cut, which
    # determines how many leading positions to mask to -100 in Step 4.
    # Example: full_ids_full has 25 tokens, max_length=20 → n_dropped=5
    #          if prompt had 8 tokens: n_prompt_remaining = max(0, 8-5) = 3
    #          full_ids = full_ids_full[5:]  (rightmost 20 tokens kept)
    if len(full_ids_full) > cfg.max_length:
        n_dropped = len(full_ids_full) - cfg.max_length
        full_ids = full_ids_full[n_dropped:]
        n_prompt_remaining = max(0, len(prompt_ids_full) - n_dropped)
    else:
        full_ids = full_ids_full
        n_prompt_remaining = len(prompt_ids_full)

    # Step 4: Build labels by masking prompt positions with -100.
    # HuggingFace cross-entropy loss ignores positions where label == -100,
    # so the loss is computed only on response tokens.
    # Example: full_ids=[1,1724,338,5132,29973,5132,338,263,...], n_prompt_remaining=5
    #          labels  =[-100,-100,-100,-100,-100, 5132, 338, 263, ...]
    labels = full_ids.copy()
    for i in range(n_prompt_remaining):
        labels[i] = -100

    n_response_tokens = len(labels) - n_prompt_remaining
    logger.debug(
        f"tokenize_prompt_response: total={len(full_ids)} tokens "
        f"(prompt={n_prompt_remaining} masked, response={n_response_tokens} active). "
        f"Caps: max_prompt={cfg.max_prompt_length}, max_length={cfg.max_length}."
    )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def pad_to_max_len(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    """Right-pad a list of token-id lists to the length of the longest sequence.

    Returns a 2-D LongTensor of shape (batch, max_len).

    Example:
        Input:  seqs=[[1, 2, 3], [4, 5], [6]], pad_value=0
        Output: tensor([[1, 2, 3],
                        [4, 5, 0],
                        [6, 0, 0]], dtype=torch.int64)
                shape=(3, 3)

    Example (label padding with -100):
        Input:  seqs=[[-100, 10, 11], [-100, -100, 12, 13]], pad_value=-100
        Output: tensor([[-100,   10,   11, -100],
                        [-100, -100,   12,   13]], dtype=torch.int64)
                shape=(2, 4)
    """
    max_len = max(len(x) for x in seqs)
    out = [x + [pad_value] * (max_len - len(x)) for x in seqs]
    return torch.tensor(out, dtype=torch.long)


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the per-token average log-probability for each sequence in a batch.

    Applies the standard auto-regressive shift (predict token t from tokens 0..t-1),
    masks positions where labels == -100 (i.e. prompt tokens), sums the remaining
    log-probs, then divides by the number of response tokens to produce a per-token
    average.  Returns a 1-D tensor of shape (batch,).

    Dividing by response length makes the SPIN margin length-agnostic: a short
    response and a long response with the same token-level quality produce the
    same scalar, so the margin comparison is fair regardless of sequence length.

    Example:
        Input:  logits shape=(2, 5, 32000)   # batch=2, seq_len=5, vocab=32000
                labels shape=(2, 5)
                labels=tensor([[-100, -100,  42, 100,  7],
                                [-100,  55,  99,   3, -100]])
                (positions with -100 are prompt tokens; others are response tokens)

        Processing:
          - shift: logits[:,:-1,:] vs labels[:,1:]  → aligned for next-token prediction
          - log_softmax applied over vocab dimension
          - gather log-prob of the actual next token at each response position
          - mask out prompt positions (label == -100)
          - sum per sequence, divide by number of response tokens

        Output: tensor([-1.6077, -1.0351])  # per-token average; one scalar per sequence
                (more negative = lower average probability per response token)

    Example (single sequence, 3 response tokens after shift):
        Input:  logits shape=(1, 4, 100), labels=tensor([[10, 20, 30, 40]])
        Output: tensor([-0.8172])  # average log-prob over tokens 20, 30, 40
    """
    batch, seq_len, vocab = logits.shape
    logger.debug(
        f"sequence_logprob_from_logits: input logits=({batch}, {seq_len}, {vocab}), "
        f"labels=({batch}, {seq_len}) — shifting by 1 for autoregressive alignment."
    )

    # Step 1: Autoregressive shift — align logits[t] with labels[t+1] so each position
    # predicts the *next* token. Drop the last logit (nothing to predict after it) and
    # the first label (no logit predicts the very first token).
    # Example: logits shape (2,5,V), labels=[-100,-100,42,100,7]
    #          shift_logits shape (2,4,V), shift_labels=[-100,42,100,7]
    #          → shift_logits[0] now predicts token 42, shift_logits[1] predicts 100, etc.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Step 2: Convert raw logits to log-probabilities over the vocabulary.
    # log_softmax normalizes scores so they sum to 1 in probability space (log scale).
    # Example: logits[t] = [2.1, 0.5, -1.3, ...]  →  log_probs[t] = [-0.42, -2.02, -3.82, ...]
    #          The highest-scored token gets the least-negative log-prob.
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Step 3: Pick the log-prob of the token that actually appeared at each position.
    # torch.gather can't index with -100, so prompt positions are temporarily set to 0
    # (a valid vocab index) before gathering — their values are zeroed out in Step 4.
    # Example: shift_labels=[-100,42,100,7], safe_labels=[0,42,100,7]
    #          token_logps[i] = log_probs[i, safe_labels[i]]
    #          → one scalar per position: [-X, log_p(42), log_p(100), log_p(7)]
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    token_logps = torch.gather(
        log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Step 4: Zero out prompt positions (label == -100) so they don't contribute to the sum.
    # Example: shift_labels=[-100,42,100,7]  →  response_mask=[False,True,True,True]
    #          token_logps before mask: [-X,    -1.2, -0.8, -2.1]
    #          token_logps after mask:  [ 0.0,  -1.2, -0.8, -2.1]  ← prompt position zeroed
    response_mask = (shift_labels != -100)
    token_logps = token_logps * response_mask

    # Step 5: Sum response-token log-probs and divide by response token count.
    # The division yields a per-token average so the SPIN margin compares quality
    # rather than sequence length — a short response with 3 tokens and a long one
    # with 30 tokens are on the same scale.
    # Example: token_logps=[0.0,-1.2,-0.8,-2.1], resp_lens=3
    #          seq_logps = (-1.2 + -0.8 + -2.1) / 3 = -1.37
    resp_lens = response_mask.sum(dim=-1)
    seq_logps = token_logps.sum(dim=-1) / resp_lens.clamp(min=1).float()

    logger.debug(
        f"  response token counts per sequence: min={resp_lens.min().item()}, "
        f"max={resp_lens.max().item()}, mean={resp_lens.float().mean().item():.1f} "
        f"(prompt positions masked with -100)."
    )
    logger.debug(
        f"  sequence log-probs (per-token avg): mean={seq_logps.mean().item():.4f}, "
        f"min={seq_logps.min().item():.4f}, max={seq_logps.max().item():.4f}."
    )
    return seq_logps


def model_sequence_logprob(model, input_ids, attention_mask, labels):
    """Run a forward pass and return sequence log-probs via sequence_logprob_from_logits.

    use_cache=False prevents KV-cache allocation during log-prob scoring, which
    is unnecessary (no generation) and wastes GPU memory.

    Example:
        Input:  model=<LlamaForCausalLM on cuda:0>,
                input_ids     shape=(4, 128) dtype=torch.int64 on cuda:0,
                attention_mask shape=(4, 128) dtype=torch.int64 on cuda:0,
                labels        shape=(4, 128), first 20 positions per row are -100 (prompt mask)

        Processing:
          - model forward pass produces logits shape=(4, 128, 32000)
          - sequence_logprob_from_logits sums log-probs over response positions

        Output: tensor([-12.43, -9.87, -15.02, -11.56], device="cuda:0")
                shape=(4,) — one scalar per sequence in the batch

    Example (batch of 1, short sequence):
        Input:  input_ids shape=(1, 10), labels=[[-100, -100, 5, 8, 12, 3, -100, -100, -100, -100]]
        Output: tensor([-3.21], device=<same as input>)
    """
    batch, seq_len = input_ids.shape
    logger.debug(
        f"model_sequence_logprob: forward pass — "
        f"input_ids=({batch}, {seq_len}), device={input_ids.device}, use_cache=False."
    )
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask, use_cache=False)
    logger.debug(
        f"  logits shape: {tuple(outputs.logits.shape)} "
        f"(batch={batch}, seq={seq_len}, vocab={outputs.logits.shape[-1]})."
    )
    seq_logps = sequence_logprob_from_logits(outputs.logits, labels)
    logger.debug(
        f"  output sequence log-probs: shape={tuple(seq_logps.shape)}, "
        f"mean={seq_logps.mean().item():.4f}, "
        f"min={seq_logps.min().item():.4f}, max={seq_logps.max().item():.4f}."
    )
    return seq_logps


@torch.no_grad()
def generate_synthetic_responses(model, tokenizer, rows: List[Dict[str, str]], cfg: SPINConfig) -> List[Dict[str, str]]:
    """Generate synthetic (rejected) responses for each row using the current model.

    Called once per SPIN iteration to produce the 'rejected' side of the training
    pairs.  The model (π_prev, frozen from the previous iteration) generates
    completions that the next iteration will learn to surpass.

    Prompts are batched in chunks of cfg.generation_batch_size to bound GPU memory.
    Only the newly generated tokens are decoded — input_lengths derived from the
    attention_mask is used to slice off the prompt portion of each output sequence.

    Returns a list of dicts with keys: prompt, response (human), synthetic_response.

    Example:
        Input:  rows=[
                    {"prompt": "What is Python?",    "response": "Python is a high-level language."},
                    {"prompt": "Explain recursion.",  "response": "A function that calls itself."},
                ]
                cfg.generation_batch_size=2, cfg.generation_max_new_tokens=200,
                cfg.generation_do_sample=True, cfg.generation_temperature=0.7

        Output: [
            {
                "prompt":            "What is Python?",
                "response":          "Python is a high-level language.",
                "synthetic_response": "Python is a general-purpose scripting language used widely..."
            },
            {
                "prompt":            "Explain recursion.",
                "response":          "A function that calls itself.",
                "synthetic_response": "Recursion refers to the process where a problem is solved..."
            },
        ]

    Note: synthetic_response is the model's own generation — it will differ from
    the human `response` and typically be lower quality early in training.
    """
    model.eval()
    out_rows = []
    bs = cfg.generation_batch_size

    # Drop rows whose formatted prompt exceeds max_prompt_length instead of
    # truncating them. Left-truncation of a chat-templated prompt amputates the
    # template header (e.g. "<|im_start|>user"), producing a malformed prompt —
    # and training would then score the synthetic response under a DIFFERENT
    # (longer, max_length-truncated) prompt than the one it was generated from.
    # Dropping here is safe: the returned rows are the source of truth for the
    # ref-logprob and training steps, so all three stages stay row-aligned.
    kept_rows = []
    n_dropped = 0
    for r in rows:
        prompt_text = build_prompt_text(r["prompt"], tokenizer, cfg)
        n_tok = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        if n_tok <= cfg.max_prompt_length:
            kept_rows.append(r)
        else:
            n_dropped += 1
    if n_dropped:
        logger.warning(
            f"Dropped {n_dropped}/{len(rows)} rows whose formatted prompt exceeds "
            f"max_prompt_length={cfg.max_prompt_length} tokens (prevents template-"
            f"mangling truncation and generation/training prompt mismatch).")
    rows = kept_rows
    if not rows:
        raise ValueError(
            f"All rows dropped: every formatted prompt exceeds "
            f"max_prompt_length={cfg.max_prompt_length}. Raise max_prompt_length "
            f"or use a dataset with shorter prompts.")

    logger.info(
        f"Starting synthetic generation: {len(rows)} rows, batch_size={bs}.")

    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        prompts = [build_prompt_text(
            r["prompt"], tokenizer, cfg) for r in chunk]
        logger.info(
            f"Batch {start // bs + 1}: processing examples {start}–{start + len(chunk) - 1} ({len(chunk)} prompts).")

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logger.info(
            f"Prompts tokenized and moved to {model.device}. Running model.generate (max_new_tokens={cfg.generation_max_new_tokens}).")

        outputs = model.generate(
            **enc,
            max_new_tokens=cfg.generation_max_new_tokens,
            do_sample=cfg.generation_do_sample,
            temperature=cfg.generation_temperature if cfg.generation_do_sample else None,
            top_p=cfg.generation_top_p if cfg.generation_do_sample else None,
            top_k=cfg.generation_top_k if cfg.generation_do_sample and cfg.generation_top_k > 0 else None,
            num_beams=cfg.generation_num_beams,
            repetition_penalty=cfg.generation_repetition_penalty,
            use_cache=cfg.generation_use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        logger.info(
            f"model.generate done. Decoding {len(outputs)} sequences (slicing off prompt tokens).")

        padded_prompt_len = enc["input_ids"].shape[1]
        for i, seq in enumerate(outputs):
            gen_ids = seq[padded_prompt_len:]
            synthetic = tokenizer.decode(
                gen_ids, skip_special_tokens=True).strip()
            out_rows.append({
                "prompt": chunk[i]["prompt"],
                "response": chunk[i]["response"],
                "synthetic_response": synthetic,
            })
        logger.info(
            f"Batch decoded and appended. Running total: {len(out_rows)} rows.")

    return out_rows


# -----------------------------
# Training args helpers
# -----------------------------

def get_iteration_lambda(cfg: SPINConfig, iteration: int) -> float:
    """Return the SPIN λ (margin regularisation weight) for a given iteration.

    If cfg.final_iteration_lambda_only is set and this is the last iteration,
    returns cfg.lambda_final_iteration so a stronger regularisation can be applied
    on the final alignment pass without affecting earlier training dynamics.

    Example (early iteration, lambda_initial used):
        Input:  cfg.num_iterations=3, cfg.lambda_initial=0.1,
                cfg.final_iteration_lambda_only=True, cfg.lambda_final_iteration=0.5,
                iteration=1
        Output: 0.1  (not the final iteration, so lambda_initial is used)

    Example (final iteration, special lambda applied):
        Input:  cfg.num_iterations=3, cfg.lambda_initial=0.1,
                cfg.final_iteration_lambda_only=True, cfg.lambda_final_iteration=0.5,
                iteration=2  (last iteration, 0-indexed)
        Output: 0.5  (lambda_final_iteration used for the last alignment pass)

    Example (final_iteration_lambda_only disabled):
        Input:  cfg.num_iterations=3, cfg.lambda_initial=0.1,
                cfg.final_iteration_lambda_only=False, iteration=2
        Output: 0.1  (lambda_initial always used)
    """
    is_final = (iteration == cfg.num_iterations - 1)
    if cfg.final_iteration_lambda_only and cfg.lambda_final_iteration is not None and is_final:
        lam = cfg.lambda_final_iteration
        logger.info(f"get_iteration_lambda(iter={iteration}): final iteration — using lambda_final_iteration={lam} "
                    f"(stronger alignment push on last pass).")
    else:
        lam = cfg.lambda_initial
        logger.info(
            f"get_iteration_lambda(iter={iteration}): using lambda_initial={lam}.")
    return lam


def get_iteration_lr(cfg: SPINConfig, iteration: int) -> float:
    """Return the learning rate for a given SPIN iteration.

    Supports a two-phase LR schedule: cfg.learning_rate for early iterations
    and cfg.learning_rate_late once cfg.late_lr_start_iteration is reached.
    Useful for decaying LR in later iterations when the model is already close
    to alignment and smaller updates prevent overshooting.

    Example (early phase):
        Input:  cfg.learning_rate=5e-5, cfg.learning_rate_late=1e-5,
                cfg.late_lr_start_iteration=2, iteration=0
        Output: 5e-5  (iteration 0 < 2, early phase)

    Example (late phase):
        Input:  cfg.learning_rate=5e-5, cfg.learning_rate_late=1e-5,
                cfg.late_lr_start_iteration=2, iteration=2
        Output: 1e-5  (iteration 2 >= 2, late phase kicks in)

    Example (no late-phase switch — late_lr_start_iteration very large):
        Input:  cfg.learning_rate=5e-5, cfg.late_lr_start_iteration=999, iteration=5
        Output: 5e-5
    """
    if iteration >= cfg.late_lr_start_iteration:
        lr = cfg.learning_rate_late
        logger.info(f"get_iteration_lr(iter={iteration}): late phase — lr={lr:.2e} "
                    f"(iteration >= late_lr_start_iteration={cfg.late_lr_start_iteration}).")
    else:
        lr = cfg.learning_rate
        logger.info(f"get_iteration_lr(iter={iteration}): early phase — lr={lr:.2e} "
                    f"({cfg.late_lr_start_iteration - iteration} iterations until late-phase switch).")
    return lr


def build_training_args(cfg: SPINConfig, iteration_dir: str, learning_rate: float, logging_dir: str) -> TrainingArguments:
    """Construct a HuggingFace TrainingArguments from SPINConfig for one iteration.

    Per-iteration output_dir (iteration_dir) keeps every iteration's checkpoints
    isolated. logging_dir is passed explicitly so TB events land under the
    consolidated tensorboard_dir rather than inside checkpoints/.
    report_to is set to [] when cfg.report_to == "none" to avoid HuggingFace
    trying to import optional logging integrations (wandb, mlflow, etc.).

    Example:
        Input:  cfg.num_epochs_per_iteration=1,
                cfg.per_device_train_batch_size=2,
                cfg.gradient_accumulation_steps=4,
                cfg.weight_decay=0.01,
                cfg.warmup_steps=50,
                cfg.lr_scheduler_type="cosine",
                cfg.logging_steps=10,
                cfg.save_strategy="no",
                cfg.save_total_limit=1,
                cfg.bf16=True, cfg.fp16=False,
                cfg.report_to="none",
                cfg.gradient_checkpointing=True,
                cfg.max_grad_norm=1.0,
                iteration_dir="output/checkpoints/iter_0/batch_000000",
                learning_rate=5e-5,
                logging_dir="output/tensorboard/iter_0/batch_000000"

        Output: TrainingArguments(
                    output_dir="output/checkpoints/iter_0/batch_000000",
                    num_train_epochs=1,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,   # effective batch = 2×4 = 8
                    learning_rate=5e-05,
                    weight_decay=0.01,
                    warmup_steps=50,
                    lr_scheduler_type="cosine",
                    logging_steps=10,
                    save_strategy="no",
                    bf16=True, fp16=False,
                    logging_dir="output/tensorboard/iter_0/batch_000000",
                    report_to=[],  # "none" → empty list so wandb is not imported
                    max_grad_norm=1.0,
                )
    """
    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    logger.info(
        "build_training_args() — constructing HuggingFace TrainingArguments...")
    logger.info(f"  output_dir:                  {iteration_dir}")
    logger.info(f"  logging_dir (TensorBoard):   {logging_dir}")
    logger.info(
        f"  num_train_epochs:            {cfg.num_epochs_per_iteration}")
    logger.info(
        f"  per_device_train_batch_size: {cfg.per_device_train_batch_size}")
    logger.info(
        f"  gradient_accumulation_steps: {cfg.gradient_accumulation_steps}  →  effective batch={eff_batch}")
    logger.info(f"  learning_rate:               {learning_rate:.2e}")
    logger.info(
        f"  lr_scheduler_type:           {cfg.lr_scheduler_type}  warmup_steps={cfg.warmup_steps}")
    logger.info(f"  weight_decay:                {cfg.weight_decay}")
    logger.info(f"  max_grad_norm:               {cfg.max_grad_norm}")
    logger.info(
        f"  bf16={cfg.bf16}, fp16={cfg.fp16}, gradient_checkpointing={cfg.gradient_checkpointing}")
    logger.info(
        f"  save_strategy={cfg.save_strategy}, save_total_limit={cfg.save_total_limit}")
    logger.info(f"  report_to:                   {cfg.report_to}")
    return TrainingArguments(
        output_dir=iteration_dir,
        num_train_epochs=cfg.num_epochs_per_iteration,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_dir=logging_dir,
        logging_strategy="steps",
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        remove_unused_columns=cfg.remove_unused_columns,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
    )


def save_json(path: str, obj: Any):
    """Serialise obj to a pretty-printed UTF-8 JSON file.

    Example:
        Input:  path="output/config.json",
                obj={"model": "llama", "iterations": 3, "lr": 5e-5}
        Output: file written at output/config.json with contents:
                {
                  "model": "llama",
                  "iterations": 3,
                  "lr": 5e-05
                }
                returns None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    """Write a list of dicts to a UTF-8 JSONL file, one JSON object per line.

    Example:
        Input:  path="cache/iter_0_batch_000000_synth.jsonl",
                rows=[
                    {"prompt": "What is Python?", "response": "A language.", "synthetic_response": "Python is..."},
                    {"prompt": "Explain AI.",      "response": "AI is...",   "synthetic_response": "Artificial..."},
                ]
        Output: file written with two lines:
                {"prompt": "What is Python?", "response": "A language.", "synthetic_response": "Python is..."}
                {"prompt": "Explain AI.", "response": "AI is...", "synthetic_response": "Artificial..."}
                returns None
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def atomic_save_jsonl(path: str, rows: List[Dict[str, Any]]):
    """Write to a .tmp file then atomically rename so a kill mid-write never leaves a corrupt cache.

    Example:
        Input:  path="cache/iter_0_batch_000000_synth.jsonl",
                rows=[{"prompt": "Hi", "synthetic_response": "Hello there."}]

        Processing:
          1. Write rows to "cache/iter_0_batch_000000_synth.jsonl.tmp"
          2. os.replace() atomically renames .tmp → .jsonl

        Output: "cache/iter_0_batch_000000_synth.jsonl" exists with correct content;
                no .tmp file remains; returns None.

        If the process is killed during step 1: the .tmp file is incomplete or absent,
        but the final .jsonl is either the previous valid version or absent —
        never a half-written file.
    """
    tmp = path + ".tmp"
    save_jsonl(tmp, rows)
    os.replace(tmp, path)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return its records as a list of dicts. Skips blank lines.

    Example:
        Input:  path="cache/iter_0_batch_000000_synth.jsonl"
                File contents:
                  {"prompt": "What is Python?", "response": "A language.", "synthetic_response": "Python is..."}
                  (blank line)
                  {"prompt": "Explain AI.", "response": "AI is...", "synthetic_response": "Artificial..."}

        Output: [
            {"prompt": "What is Python?", "response": "A language.", "synthetic_response": "Python is..."},
            {"prompt": "Explain AI.",      "response": "AI is...",   "synthetic_response": "Artificial..."},
        ]

    Example (logprobs file):
        Input:  path="cache/iter_0_batch_000000_logprobs.jsonl"
                File: {"ref_chosen_logp": -12.43, "ref_rejected_logp": -18.07}
        Output: [{"ref_chosen_logp": -12.43, "ref_rejected_logp": -18.07}]
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _log_trainable_parameters(model):
    """Log the trainable vs total parameter count and the trainable percentage.

    Example:
        Input:  model=<PeftModel wrapping LlamaForCausalLM 1B>
                (LoRA r=16 added to q_proj, k_proj, v_proj, o_proj of 16 layers)
        Output: logs "Trainable parameters: 8,388,608 / 1,236,862,976 (0.68%)"
                returns None

    Example:
        Input:  model=<LlamaForCausalLM 1B> with all parameters trainable (full fine-tune)
        Output: logs "Trainable parameters: 1,236,862,976 / 1,236,862,976 (100.00%)"
                returns None
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def make_trainable(model, cfg: SPINConfig):
    """Convert a frozen ref model into a trainable model without reloading from disk.

    With use_lora=True the base weights stay frozen; only the LoRA adapter
    parameters (a tiny fraction of the total) are made trainable.  This halves
    the memory needed for gradients and optimizer states compared to full fine-tuning.

    Example (LoRA mode):
        Input:  model=<LlamaForCausalLM 1B, all params frozen>,
                cfg.use_lora=True, cfg.lora_r=16, cfg.lora_alpha=32,
                cfg.lora_dropout=0.05,
                cfg.lora_target_modules="q_proj,k_proj,v_proj,o_proj",
                cfg.gradient_checkpointing=True

        Output: <PeftModel wrapping LlamaForCausalLM>
                - base LLaMA weights remain frozen (requires_grad=False)
                - 8.4M LoRA adapter params have requires_grad=True (~0.68% of total)
                - model.training=True
                - gradient checkpointing enabled
                - logs "Trainable parameters: 8,388,608 / 1,236,862,976 (0.68%)"

    Example (full fine-tune mode):
        Input:  model=<LlamaForCausalLM 1B, all params frozen>,
                cfg.use_lora=False, cfg.gradient_checkpointing=False

        Output: <LlamaForCausalLM> with all 1.2B params set to requires_grad=True,
                model.training=True

    Example (wrong lora_target_modules — auto-detection kicks in):
        Input:  cfg.lora_target_modules="query,value" (GPT-2 style names on a LLaMA model)
        Output: warning logged; auto-detects ["q_proj","k_proj","v_proj","o_proj"] for LLaMA
                and continues without raising
    """
    logger.info(
        "make_trainable() — converting frozen π_prev into trainable π_θ...")

    if hasattr(model, "_orig_mod"):
        logger.info(
            "  Unwrapping torch.compile OptimizedModule before PEFT/LoRA wrapping.")
        model = model._orig_mod

    if cfg.use_lora:
        target_modules = cfg.lora_target_modules.split(",")

        # Validate that target modules exist in this model; auto-detect if not.
        model_linear_names = {
            name.split(".")[-1]
            for name, mod in model.named_modules()
            if mod.__class__.__name__ in ("Linear", "Conv1D")
        }
        missing = [m for m in target_modules if m not in model_linear_names]
        if missing:
            # Well-known architecture fallbacks keyed by suffix sets present in the model.
            _ARCH_TARGETS = [
                ({"c_attn", "c_proj"}, ["c_attn", "c_proj"]),        # GPT-2 family
                ({"q_proj", "v_proj"}, ["q_proj", "k_proj", "v_proj", "o_proj"]),  # LLaMA/Mistral
                ({"query_key_value"}, ["query_key_value", "dense"]),  # Falcon/BLOOM
                ({"Wqkv"}, ["Wqkv", "out_proj"]),                     # MPT
            ]
            detected = None
            for required, modules in _ARCH_TARGETS:
                if required.issubset(model_linear_names):
                    detected = [m for m in modules if m in model_linear_names]
                    break
            if detected:
                logger.warning(
                    f"  lora_target_modules {target_modules} not found in model "
                    f"(available linear layers: {sorted(model_linear_names)}). "
                    f"Auto-detected targets for this architecture: {detected}")
                target_modules = detected
            else:
                raise ValueError(
                    f"lora_target_modules {target_modules} not found in model. "
                    f"Available linear layer names: {sorted(model_linear_names)}. "
                    f"Set lora_target_modules in your config to match your model architecture."
                )

        logger.info(
            f"  LoRA mode: freezing all base weights, adding adapters to: {target_modules}")
        logger.info(f"  LoRA config: r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
                    f"dropout={cfg.lora_dropout}, scale={cfg.lora_alpha / cfg.lora_r:.2f}")
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        logger.info(
            "  PEFT model created — base weights frozen, LoRA adapter params trainable.")

        # Keep trainable adapter params in fp32. The base model is loaded in bf16 and
        # HF's bf16=True is autocast-only (no fp32 master weights), so optimizing bf16
        # adapter params directly makes most updates vanish: bf16 has ~8 mantissa bits
        # (relative resolution ~4e-3), while AdamW updates at lr≈1e-6 are ~1e-4–1e-5
        # relative — they round to zero on every weight not near 0. PEFT's LoraLayer
        # casts activations to the adapter dtype in forward, so mixed bf16-base /
        # fp32-adapter is fully supported (the standard QLoRA configuration).
        n_upcast = 0
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
                n_upcast += 1
        if n_upcast:
            logger.info(
                f"  Upcast {n_upcast} trainable adapter tensors to fp32 "
                f"(bf16 master weights round away lr-scale updates).")

        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
            logger.info(
                "  enable_input_require_grads() called (required for grad-ckpt + PEFT).")
    else:
        logger.info(
            "  Full fine-tuning mode: all parameters set to requires_grad=True.")
        for p in model.parameters():
            p.requires_grad = True

    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("  Gradient checkpointing enabled; use_cache=False.")
    else:
        model.config.use_cache = cfg.generation_use_cache

    if cfg.log_trainable_parameters:
        _log_trainable_parameters(model)

    if cfg.compile_model:
        logger.info(f"  Compiling train_model with torch.compile "
                    f"(backend={cfg.compile_backend}, mode={cfg.compile_mode}, "
                    f"dynamic={cfg.compile_dynamic}, fullgraph={cfg.compile_fullgraph})...")
        model = maybe_compile_model(model, cfg, label="train_model")

    logger.info("make_trainable() complete — model ready for SPINTrainer.")
    return model


def merge_lora_and_get_base(model, cfg: SPINConfig):
    """Merge LoRA adapter weights into the base model and return the unwrapped model.

    Called before save_pretrained so the checkpoint written to disk is a plain
    AutoModelForCausalLM with no PEFT dependency.  The next SPIN iteration then
    loads it with load_causal_lm() exactly like any other checkpoint.

    Handles the case where the model was wrapped by torch.compile: the compiled
    wrapper stores the original module at ._orig_mod, which is unwrapped first so
    PeftModel.merge_and_unload() can operate on the underlying PEFT model.

    Example (LoRA mode):
        Input:  model=<PeftModel wrapping LlamaForCausalLM> (8.4M LoRA adapter params),
                cfg.use_lora=True
        Output: <LlamaForCausalLM> with LoRA deltas folded into the original weight matrices;
                no longer a PeftModel — can be loaded with AutoModelForCausalLM.from_pretrained()
                logs "LoRA adapters merged into base model weights."

    Example (torch.compile wrapper):
        Input:  model=<OptimizedModule._orig_mod=<PeftModel>>, cfg.use_lora=True
        Output: compile wrapper unwrapped first, then LoRA merged;
                returns plain <LlamaForCausalLM>

    Example (full fine-tune, no LoRA):
        Input:  model=<LlamaForCausalLM>, cfg.use_lora=False
        Output: same <LlamaForCausalLM> returned unchanged (early return, no merge)
    """
    if not cfg.use_lora:
        return model
    # Unwrap torch.compile() wrapper so PeftModel isinstance check succeeds.
    if hasattr(model, "_orig_mod"):
        logger.info("Unwrapping torch.compile wrapper before LoRA merge.")
        model = model._orig_mod
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged into base model weights.")

    return model


def free_model(model):
    """Delete a model reference, run garbage collection, and empty the CUDA cache.

    Called between SPIN iterations after trainer.save_model() completes.  Each
    checkpoint can be several GB; freeing before loading the next iteration's model
    prevents OOM when total GPU memory is close to the model size.

    Example:
        Input:  model=<LlamaForCausalLM 1B on cuda:0>  (~2 GB VRAM allocated)
        Output: model deleted, gc.collect() run, torch.cuda.empty_cache() called;
                VRAM drops from ~2 GB back to near-zero; returns None

    Example (no CUDA available):
        Input:  model=<LlamaForCausalLM on CPU>
        Output: model deleted, gc.collect() run; no CUDA ops performed; returns None
    """
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def compute_ref_logprobs(model, tokenizer, rows: List[Dict[str, str]], cfg: SPINConfig) -> List[Dict[str, float]]:
    """Score every (chosen, rejected) pair under the frozen reference model.

    Must be called while the model is still in its frozen state (before
    make_trainable() converts it to the trainable π_θ).  The resulting log-probs
    are stored alongside each training row so the SPINDataset / SPINTrainer can
    compute the SPIN margin loss without a second reference-model forward pass
    during training, saving both memory and time.

    Processes rows in batches of cfg.ref_logprob_batch_size for GPU efficiency.
    Sequences within a batch are right-padded to the batch maximum length;
    padding positions carry label -100 so they are masked out of the log-prob sum.

    Returns a list of dicts with keys: ref_chosen_logp, ref_rejected_logp.

    Example:
        Input:  rows=[
                    {
                        "prompt": "What is Python?",
                        "response": "Python is a high-level language.",
                        "synthetic_response": "Python is a general-purpose scripting language..."
                    },
                    {
                        "prompt": "Explain recursion.",
                        "response": "A function that calls itself.",
                        "synthetic_response": "Recursion is the process where a function..."
                    },
                ]
                cfg.ref_logprob_batch_size=2

        Output: [
            {"ref_chosen_logp": -12.43, "ref_rejected_logp": -18.07},
            {"ref_chosen_logp":  -9.82, "ref_rejected_logp": -14.55},
        ]
        (chosen human responses typically have higher log-prob than synthetic ones
         under a good reference model, so ref_chosen_logp > ref_rejected_logp)
    """
    bs = cfg.ref_logprob_batch_size
    pad_id = tokenizer.pad_token_id
    logger.info(
        f"Computing reference log-probs for {len(rows)} rows (batch_size={bs})...")
    model.eval()
    ref_logprobs = []

    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]

        # Step 1: Tokenize every (prompt, chosen_response) and (prompt, rejected_response)
        # pair in the chunk. Each call returns input_ids, attention_mask, and labels
        # with prompt positions masked to -100.
        # Example: chunk has 2 rows → chosen_tok and rejected_tok each have 2 dicts.
        chosen_tok = [tokenize_prompt_response(
            tokenizer, r["prompt"], r["response"],           cfg) for r in chunk]
        rejected_tok = [tokenize_prompt_response(
            tokenizer, r["prompt"], r["synthetic_response"], cfg) for r in chunk]

        # Step 2: Right-pad all sequences in the batch to the same length.
        # Chosen and rejected are padded independently (they may have different max lengths).
        # pad_value=pad_id for input_ids, 0 for attention_mask (ignore padding), -100 for labels.
        # Example: chosen lengths=[15,20] → both padded to 20; rejected lengths=[18,12] → both 18.
        chosen_ids = pad_to_max_len(
            [x["input_ids"] for x in chosen_tok],   pad_id).to(model.device)
        chosen_mask = pad_to_max_len(
            [x["attention_mask"] for x in chosen_tok],   0).to(model.device)
        chosen_labels = pad_to_max_len(
            [x["labels"] for x in chosen_tok],   -100).to(model.device)

        rejected_ids = pad_to_max_len(
            [x["input_ids"] for x in rejected_tok], pad_id).to(model.device)
        rejected_mask = pad_to_max_len(
            [x["attention_mask"] for x in rejected_tok], 0).to(model.device)
        rejected_labels = pad_to_max_len(
            [x["labels"] for x in rejected_tok], -100).to(model.device)

        # Step 3: Forward pass under the frozen reference model (no_grad is set by decorator).
        # Returns one per-token-average log-prob scalar per sequence.
        # Example: chosen_logps=[-12.43, -9.82], rejected_logps=[-18.07, -14.55]
        chosen_logps = model_sequence_logprob(
            model, chosen_ids,   chosen_mask,   chosen_labels)
        rejected_logps = model_sequence_logprob(
            model, rejected_ids, rejected_mask, rejected_labels)

        # Step 4: Store the scalar pair for each row so SPINTrainer can compute the
        # SPIN margin loss during training without a second reference-model forward pass.
        for c_lp, r_lp in zip(chosen_logps.tolist(), rejected_logps.tolist()):
            ref_logprobs.append(
                {"ref_chosen_logp": c_lp, "ref_rejected_logp": r_lp})

        logger.info(
            f"  ref_logprob: {min(start + bs, len(rows))}/{len(rows)} rows scored.")

    logger.info(f"Reference log-probs computed for {len(ref_logprobs)} rows.")
    return ref_logprobs

# ── Per-batch file paths ──────────────────────────────────────────────────────


def synth_path(cfg, iteration, k):
    """Return the JSONL path for the synthetic responses of batch k in a given iteration.

    Example:
        Input:  cfg.synthetic_cache_dir="output/synth_cache", iteration=1, k=3
        Output: "output/synth_cache/iter_1_batch_000003_synth.jsonl"
    """
    return os.path.join(cfg.synthetic_cache_dir,
                        f"iter_{iteration}_batch_{k:06d}_synth.jsonl")


def logprobs_path(cfg, iteration, k):
    """Return the JSONL path for the reference log-probs of batch k in a given iteration.

    Example:
        Input:  cfg.synthetic_cache_dir="output/synth_cache", iteration=0, k=0
        Output: "output/synth_cache/iter_0_batch_000000_logprobs.jsonl"
    """
    return os.path.join(cfg.synthetic_cache_dir,
                        f"iter_{iteration}_batch_{k:06d}_logprobs.jsonl")


def tokenization_fingerprint(cfg) -> str:
    """Return a short hash of every config field that changes tokenization output.

    Embedded in tokenized-cache filenames so that switching the model/tokenizer,
    sequence caps, or chat-template settings invalidates the cache instead of
    silently loading tensors produced under the old settings (the caches were
    previously keyed by iteration/batch index only).

    Example:
        Input:  cfg.model_name_or_path="Qwen/Qwen2.5-0.5B", cfg.max_length=512,
                cfg.max_prompt_length=256, cfg.chat_template_mode="auto", ...
        Output: "3f9a1c2e"  (stable 8-hex-char digest; same cfg → same digest)
    """
    key = "|".join(str(v) for v in (
        cfg.model_name_or_path,
        cfg.tokenizer_name_or_path,
        cfg.max_length,
        cfg.max_prompt_length,
        cfg.truncation_side,
        cfg.chat_template_mode,
        cfg.instruction_prefix,
        cfg.response_prefix,
        cfg.add_eos_to_response,
    ))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def tokenized_path(cfg, iteration, k):
    """Return the .pt cache path for pre-tokenized tensors of batch k in a given iteration.

    The filename embeds tokenization_fingerprint(cfg) so caches from different
    model/tokenizer/sequence settings never collide.

    Example:
        Input:  cfg.synthetic_cache_dir="output/synth_cache", iteration=2, k=10
        Output: "output/synth_cache/iter_2_batch_000010_tokenized_3f9a1c2e.pt"
    """
    return os.path.join(cfg.synthetic_cache_dir,
                        f"iter_{iteration}_batch_{k:06d}_tokenized_{tokenization_fingerprint(cfg)}.pt")


def batch_train_dir(cfg, iteration, k):
    """Return the directory where the merged model is saved after training batch k.

    Example:
        Input:  cfg.checkpoints_dir="output/checkpoints", iteration=1, k=5
        Output: "output/checkpoints/iter_1/batch_000005"
    """
    return os.path.join(cfg.checkpoints_dir, f"iter_{iteration}", f"batch_{k:06d}")


def batch_done_path(cfg, iteration, k):
    """Return the path of the .done sentinel file that marks a fully completed training batch.

    Example:
        Input:  cfg.checkpoints_dir="output/checkpoints", iteration=0, k=2
        Output: "output/checkpoints/iter_0/batch_000002/.done"
    """
    return os.path.join(batch_train_dir(cfg, iteration, k), ".done")


def file_valid(path):
    """Return True if the file exists and is non-empty, False otherwise.

    Example:
        Input:  path="output/synth_cache/iter_0_batch_000000_synth.jsonl"  (exists, 4 KB)
        Output: True

        Input:  path="output/synth_cache/iter_0_batch_000000_synth.jsonl"  (does not exist)
        Output: False

        Input:  path="output/synth_cache/iter_0_batch_000000_synth.jsonl"  (exists but 0 bytes — corrupt write)
        Output: False
    """
    return os.path.exists(path) and os.path.getsize(path) > 0
