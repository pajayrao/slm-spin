import os
import gc
import json
import argparse
import logging
import glob
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
    "format":'%(asctime)s %(levelname)-8s %(message)s',
    "level":logging.INFO,
    "datefmt":'%Y-%m-%d %H:%M:%S',
}

logging.basicConfig(**logging_kwargs)
logger = logging.getLogger(__name__)


def pre_start_cleanup():
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    for lock_file in glob.glob(os.path.join(hf_cache, "**", "*.lock"), recursive=True):
        try:
            os.remove(lock_file)
            logger.info(f"Removed stale lock: {lock_file}", flush=True)
        except OSError:
            pass

pre_start_cleanup()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_memory(tag: str):
        
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
        logger.info(f"[MEM {tag}] CPU RSS {rss_mb:.0f} MB | GPU alloc {alloc_mb:.0f} MB | GPU reserved {reserved_mb:.0f} MB")
    else:
        logger.info(f"[MEM {tag}] CPU RSS {rss_mb:.0f} MB")

def str2dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")

def parse_args() -> SPINConfig:
    parser = argparse.ArgumentParser(description="SPIN training")

    for field_name, field_def in SPINConfig.__dataclass_fields__.items():
        default = field_def.default
        if isinstance(default, bool):
            parser.add_argument(f"--{field_name}", type=str, default=str(default))
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
    if cfg.chat_template_mode == "plain":
        return user_prompt
    if cfg.chat_template_mode == "instruction_response":
        return f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"
    if cfg.chat_template_mode == "auto":
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"
        return f"{cfg.instruction_prefix}{user_prompt}{cfg.response_prefix}"
    raise ValueError(f"Unknown chat_template_mode: {cfg.chat_template_mode}")

def normalize_chat_dataset_record(example):
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

def load_base_dataset_fixed(dataset_name=None, dataset_config_name=None, split="train_sft", data_path=None, limit=None):
    if data_path:
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            ds = load_dataset("json", data_files=data_path, split="train")
        elif data_path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=data_path, split="train")
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
    else:
        ds = load_dataset(dataset_name, dataset_config_name, split=split)

    rows = []
    skipped = 0

    for ex in ds:
        item = normalize_chat_dataset_record(ex)
        if item is None:
            skipped += 1
            continue
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            break

    logger.info(f"Loaded rows: {len(rows)}")
    logger.info(f"Skipped rows: {skipped}")

    if len(rows) == 0:
        raise ValueError("No valid prompt/response pairs found.")

    return HFDataset.from_list(rows)


# -----------------------------
# Model loading
# -----------------------------

def load_tokenizer(cfg: SPINConfig):
    tok_name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = cfg.truncation_side
    tokenizer.padding_side = "left"
    return tokenizer

def load_causal_lm(model_path: str, cfg: SPINConfig, trainable: bool = True):
    kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=str2dtype(cfg.torch_dtype),
    )
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if trainable and cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = cfg.generation_use_cache

    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    if cfg.log_trainable_parameters:
        _log_trainable_parameters(model)

    return model


# -----------------------------
# Tokenization
# -----------------------------

def build_prompt_text(prompt: str, tokenizer, cfg: SPINConfig) -> str:
    return maybe_apply_chat_template(tokenizer, prompt, cfg)

def build_full_text(prompt: str, response: str, tokenizer, cfg: SPINConfig) -> str:
    txt = build_prompt_text(prompt, tokenizer, cfg) + response
    if cfg.add_eos_to_response and tokenizer.eos_token and not txt.endswith(tokenizer.eos_token):
        txt += tokenizer.eos_token
    return txt

def tokenize_prompt_response(tokenizer, prompt: str, response: str, cfg: SPINConfig) -> Dict[str, Any]:
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
        prompt_ids = prompt_ids[:len(full_ids)]

    labels = full_ids.copy()
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

def pad_to_max_len(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    out = [x + [pad_value] * (max_len - len(x)) for x in seqs]
    return torch.tensor(out, dtype=torch.long)


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * (shift_labels != -100)
    return token_logps.sum(dim=-1)

def model_sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return sequence_logprob_from_logits(outputs.logits, labels)

@torch.no_grad()
def generate_synthetic_responses(model, tokenizer, rows: List[Dict[str, str]], cfg: SPINConfig) -> List[Dict[str, str]]:
    model.eval()
    out_rows = []
    bs = cfg.generation_batch_size
    logger.info("generate_synthetic_responses ============================ 1 ==================================")

    for start in range(0, len(rows), bs):
        chunk = rows[start:start + bs]
        prompts = [build_prompt_text(r["prompt"], tokenizer, cfg) for r in chunk]
        logger.info("generate_synthetic_responses ============================ 2 ==================================")

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logger.info("generate_synthetic_responses ============================ 3 ==================================")

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
        logger.info("generate_synthetic_responses ============================ 4 ==================================")

        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        for i, seq in enumerate(outputs):
            gen_ids = seq[input_lengths[i]:]
            synthetic = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            out_rows.append({
                "prompt": chunk[i]["prompt"],
                "response": chunk[i]["response"],
                "synthetic_response": synthetic,
            })
        logger.info("generate_synthetic_responses ============================ 4 ==================================")

    return out_rows


# -----------------------------
# Training args helpers
# -----------------------------

def get_iteration_lambda(cfg: SPINConfig, iteration: int) -> float:
    if cfg.final_iteration_lambda_only and cfg.lambda_final_iteration is not None and iteration == cfg.num_iterations - 1:
        return cfg.lambda_final_iteration
    return cfg.lambda_initial

def get_iteration_lr(cfg: SPINConfig, iteration: int) -> float:
    if iteration >= cfg.late_lr_start_iteration:
        return cfg.learning_rate_late
    return cfg.learning_rate

def build_training_args(cfg: SPINConfig, iteration_dir: str, learning_rate: float) -> TrainingArguments:
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
        logging_dir=os.path.join(iteration_dir, "tb_logs"),   
        logging_strategy="steps",                            
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        remove_unused_columns=cfg.remove_unused_columns,
        dataloader_num_workers=cfg.dataloader_num_workers,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        deepspeed=cfg.deepspeed,
    )

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _log_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def make_trainable(model, cfg: SPINConfig):
    """Convert a frozen ref model into a trainable model without reloading from disk.

    With use_lora=True the base weights stay frozen; only the LoRA adapter
    parameters (a tiny fraction of the total) are made trainable.  This halves
    the memory needed for gradients and optimizer states compared to full fine-tuning.
    """
    if cfg.use_lora:
        # Base weights must be frozen before PEFT wraps them; PEFT then enables
        # only the adapter parameters it inserts.
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        # Required when using gradient checkpointing with PEFT so the first
        # layer's input tensor tracks gradients even though it isn't a leaf.
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = cfg.generation_use_cache
    if cfg.log_trainable_parameters:
        _log_trainable_parameters(model)
    return model


def merge_lora_and_get_base(model, cfg: SPINConfig):
    """Merge LoRA adapter weights into the base model and return the unwrapped model.

    Called before save_pretrained so the checkpoint written to disk is a plain
    AutoModelForCausalLM with no PEFT dependency.  The next SPIN iteration then
    loads it with load_causal_lm() exactly like any other checkpoint.
    """
    if not cfg.use_lora:
        return model
    try:
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            logger.info("LoRA adapters merged into base model weights.")
    except ImportError:
        logger.warning("peft not installed; skipping LoRA merge.")
    return model


def free_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def compute_ref_logprobs(model, tokenizer, rows: List[Dict[str, str]], cfg: SPINConfig) -> List[Dict[str, float]]:
    model.eval()
    ref_logprobs = []
    for row in rows:
        chosen = tokenize_prompt_response(tokenizer, row["prompt"], row["response"], cfg)
        rejected = tokenize_prompt_response(tokenizer, row["prompt"], row["synthetic_response"], cfg)

        chosen_ids = torch.tensor([chosen["input_ids"]], dtype=torch.long).to(model.device)
        chosen_mask = torch.tensor([chosen["attention_mask"]], dtype=torch.long).to(model.device)
        chosen_labels = torch.tensor([chosen["labels"]], dtype=torch.long).to(model.device)

        rejected_ids = torch.tensor([rejected["input_ids"]], dtype=torch.long).to(model.device)
        rejected_mask = torch.tensor([rejected["attention_mask"]], dtype=torch.long).to(model.device)
        rejected_labels = torch.tensor([rejected["labels"]], dtype=torch.long).to(model.device)

        ref_logprobs.append({
            "ref_chosen_logp": model_sequence_logprob(model, chosen_ids, chosen_mask, chosen_labels).item(),
            "ref_rejected_logp": model_sequence_logprob(model, rejected_ids, rejected_mask, rejected_labels).item(),
        })
    return ref_logprobs

