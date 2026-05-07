import os
import gc
import json
import argparse
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
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    for lock_file in glob.glob(os.path.join(hf_cache, "**", "*.lock"), recursive=True):
        try:
            os.remove(lock_file)
            logger.info(f"Removed stale lock: {lock_file}", flush=True)
        except OSError:
            pass


pre_start_cleanup()


def ensure_dir(path: str):
    """Create a directory (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def log_memory(tag: str):
    """Log current CPU RSS and GPU allocated/reserved memory to the Python logger.

    Args:
        tag: A short label (e.g. "before_trainer_init") printed alongside the numbers
             so spikes can be correlated with specific code events in the log.
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
                result = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
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
    """Return the formatted prompt string (chat template applied, no response appended)."""
    return maybe_apply_chat_template(tokenizer, prompt, cfg)


def build_full_text(prompt: str, response: str, tokenizer, cfg: SPINConfig) -> str:
    """Concatenate the formatted prompt and response into one string for tokenization.

    Appends eos_token when cfg.add_eos_to_response is True and the response
    doesn't already end with it, so the model learns to terminate cleanly.
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
    """
    prompt_text = build_prompt_text(prompt, tokenizer, cfg)
    full_text = build_full_text(prompt, response, tokenizer, cfg)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_prompt_length,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_length,
    )["input_ids"]

    if len(full_ids) < len(prompt_ids):
        logger.debug(
            f"tokenize_prompt_response: full_ids ({len(full_ids)}) shorter than prompt_ids "
            f"({len(prompt_ids)}) — truncating prompt_ids to match (truncation_side={cfg.truncation_side})."
        )
        prompt_ids = prompt_ids[:len(full_ids)]

    labels = full_ids.copy()
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    n_prompt_tokens = min(len(prompt_ids), len(labels))
    n_response_tokens = len(labels) - n_prompt_tokens
    logger.debug(
        f"tokenize_prompt_response: total={len(full_ids)} tokens "
        f"(prompt={n_prompt_tokens} masked, response={n_response_tokens} active). "
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
    """
    max_len = max(len(x) for x in seqs)
    out = [x + [pad_value] * (max_len - len(x)) for x in seqs]
    return torch.tensor(out, dtype=torch.long)


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the sum of per-token log-probabilities for each sequence in a batch.

    Applies the standard auto-regressive shift (predict token t from tokens 0..t-1),
    masks positions where labels == -100 (i.e. prompt tokens), and sums the
    remaining log-probs.  Returns a 1-D tensor of shape (batch,).
    """
    batch, seq_len, vocab = logits.shape
    logger.debug(
        f"sequence_logprob_from_logits: input logits=({batch}, {seq_len}, {vocab}), "
        f"labels=({batch}, {seq_len}) — shifting by 1 for autoregressive alignment."
    )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    token_logps = torch.gather(
        log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    response_mask = (shift_labels != -100)
    token_logps = token_logps * response_mask

    # Per-sequence response-token count (non-masked positions = actual response tokens).
    resp_lens = response_mask.sum(dim=-1)
    seq_logps = token_logps.sum(dim=-1)

    logger.debug(
        f"  response token counts per sequence: min={resp_lens.min().item()}, "
        f"max={resp_lens.max().item()}, mean={resp_lens.float().mean().item():.1f} "
        f"(prompt positions masked with -100)."
    )
    logger.debug(
        f"  sequence log-probs: mean={seq_logps.mean().item():.4f}, "
        f"min={seq_logps.min().item():.4f}, max={seq_logps.max().item():.4f} "
        f"(sum of per-token log-probs over response tokens only)."
    )
    return seq_logps


def model_sequence_logprob(model, input_ids, attention_mask, labels):
    """Run a forward pass and return sequence log-probs via sequence_logprob_from_logits.

    use_cache=False prevents KV-cache allocation during log-prob scoring, which
    is unnecessary (no generation) and wastes GPU memory.
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
    """
    model.eval()
    out_rows = []
    bs = cfg.generation_batch_size
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

        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        for i, seq in enumerate(outputs):
            gen_ids = seq[input_lengths[i]:]
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
        deepspeed=cfg.deepspeed,
    )


def save_json(path: str, obj: Any):
    """Serialise obj to a pretty-printed UTF-8 JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    """Write a list of dicts to a UTF-8 JSONL file, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def atomic_save_jsonl(path: str, rows: List[Dict[str, Any]]):
    """Write to a .tmp file then atomically rename so a kill mid-write never leaves a corrupt cache."""
    tmp = path + ".tmp"
    save_jsonl(tmp, rows)
    os.replace(tmp, path)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return its records as a list of dicts. Skips blank lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _log_trainable_parameters(model):
    """Log the trainable vs total parameter count and the trainable percentage."""
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
    """
    logger.info(
        "make_trainable() — converting frozen π_prev into trainable π_θ...")

    if hasattr(model, "_orig_mod"):
        logger.info(
            "  Unwrapping torch.compile OptimizedModule before PEFT/LoRA wrapping.")
        model = model._orig_mod

    if cfg.use_lora:
        target_modules = cfg.lora_target_modules.split(",")
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
    """
    bs = cfg.ref_logprob_batch_size
    pad_id = tokenizer.pad_token_id
    logger.info(
        f"Computing reference log-probs for {len(rows)} rows (batch_size={bs})...")
    model.eval()
    ref_logprobs = []

    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]

        chosen_tok = [tokenize_prompt_response(
            tokenizer, r["prompt"], r["response"],           cfg) for r in chunk]
        rejected_tok = [tokenize_prompt_response(
            tokenizer, r["prompt"], r["synthetic_response"], cfg) for r in chunk]

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

        chosen_logps = model_sequence_logprob(
            model, chosen_ids,   chosen_mask,   chosen_labels)
        rejected_logps = model_sequence_logprob(
            model, rejected_ids, rejected_mask, rejected_labels)

        for c_lp, r_lp in zip(chosen_logps.tolist(), rejected_logps.tolist()):
            ref_logprobs.append(
                {"ref_chosen_logp": c_lp, "ref_rejected_logp": r_lp})

        logger.info(
            f"  ref_logprob: {min(start + bs, len(rows))}/{len(rows)} rows scored.")

    logger.info(f"Reference log-probs computed for {len(ref_logprobs)} rows.")
    return ref_logprobs

# ── Per-batch file paths ──────────────────────────────────────────────────────


def synth_path(cfg, iteration, k):
    return os.path.join(cfg.synthetic_cache_dir,
                        f"iter_{iteration}_batch_{k:06d}_synth.jsonl")


def logprobs_path(cfg, iteration, k):
    return os.path.join(cfg.synthetic_cache_dir,
                        f"iter_{iteration}_batch_{k:06d}_logprobs.jsonl")


def batch_train_dir(cfg, iteration, k):
    return os.path.join(cfg.checkpoints_dir, f"iter_{iteration}", f"batch_{k:06d}")


def batch_done_path(cfg, iteration, k):
    return os.path.join(batch_train_dir(cfg, iteration, k), ".done")


def file_valid(path):
    return os.path.exists(path) and os.path.getsize(path) > 0
