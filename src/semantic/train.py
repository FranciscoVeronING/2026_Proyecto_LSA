"""Entrenamiento LoRA, export GGUF y evaluación de un modelo."""
from __future__ import annotations

# unsloth antes que trl/transformers/peft
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import gc
from pathlib import Path
from typing import Any, TextIO

import torch
from trl import SFTConfig, SFTTrainer

from config import (
    MAX_SEQ_LENGTH,
    TARGET_MODULES,
    TRAIN_SEQ_LENGTH,
    is_large_model,
)
from dataset import load_parallel_pairs, referencias_por_espanol, split_pairs
from metrics import evaluar_pares, load_gguf_engine
from gguf_export import export_merged_gguf, find_q4_gguf
from io_utils import log_line, save_json
from prompts import build_messages


def _release_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_lora_model(model: Any) -> Any:
    """LoRA r=16. Si fallan los nombres de capa, usa all-linear."""
    common = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    }
    try:
        return FastLanguageModel.get_peft_model(model, target_modules=TARGET_MODULES, **common)
    except ValueError as exc:
        print(f"target_modules estándar falló ({exc}). Reintentando con all-linear...")
        return FastLanguageModel.get_peft_model(model, target_modules="all-linear", **common)


def train_and_eval_single_model(
    model_cfg: dict[str, Any],
    output_dir: Path,
    dataset_path: Path,
    system_prompt: str,
    epochs: int = 5,
    max_steps: int = -1,
    log_fp: TextIO | None = None,
) -> dict[str, Any]:
    model_name = model_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log_line(log_fp, f"Cargando {model_name} en 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg.get("chat_template", "chatml"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_lora_model(model)

    pairs = load_parallel_pairs(dataset_path)
    train_raw, test_raw = split_pairs(pairs)

    def format_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        texts: list[str] = []
        for spanish, glosses in zip(examples["spanish"], examples["glosses"]):
            messages = build_messages(system_prompt, spanish, glosses)
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_dataset = train_raw.map(format_prompts, batched=True)

    large = is_large_model(model_name)
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2 if large else 4,
        "gradient_accumulation_steps": 8 if large else 4,
        "warmup_steps": 10,
        "learning_rate": 3e-4 if large else 5e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
        "dataset_text_field": "text",
        "max_length": TRAIN_SEQ_LENGTH,
        "packing": False,
        "report_to": "none",
    }
    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps
    else:
        sft_kwargs["num_train_epochs"] = epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**sft_kwargs),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=model_cfg.get("instruction_part", "<|im_start|>user\n"),
        response_part=model_cfg.get("response_part", "<|im_start|>assistant\n"),
    )

    log_line(log_fp, f"Entrenando {model_name} (español → glosas LSA)...")
    train_result = trainer.train()
    final_loss = getattr(train_result, "training_loss", None)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    gguf_dir = export_merged_gguf(model, tokenizer, output_dir, log_fp=log_fp)
    _release_cuda(model, tokenizer, trainer)

    gguf_file = find_q4_gguf(gguf_dir)
    log_line(log_fp, f"Evaluando GGUF: {gguf_file.name}")
    llm = load_gguf_engine(gguf_file)
    metrics = evaluar_pares(
        llm=llm,
        system_prompt=system_prompt,
        test_items=list(test_raw),
        refs_por_espanol=referencias_por_espanol(pairs),
        model_name=model_name,
    )
    _release_cuda(llm)

    metrics["train_loss_final"] = final_loss
    metrics["gguf_dir"] = str(gguf_dir)
    metrics["gguf_file"] = str(gguf_file)
    save_json(output_dir / "metrics.json", metrics)
    return metrics
