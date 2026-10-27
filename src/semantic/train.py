"""Train the catalog LLMs, export GGUF Q4_K_M, and evaluate the 15% holdout."""
from __future__ import annotations

# Unsloth must be imported before trl, transformers, or peft.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import argparse
import gc
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from glosses import gloss_text, matches, unify_fingerspelled_glosses
from inference import gloss_messages, load_gguf, translate_gguf
from metrics import score_all
from models import (
    DATASET_GLOSSES,
    LOGS,
    MAX_SEQ_LENGTH,
    OUTPUTS,
    PROMPT,
    model_folder,
    select_models,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TEMP_GGUF = Path("D:/temp_gguf")
TEMP_GGUF_EXTRA = Path("D:/temp_gguf_gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the catalog, export GGUF, and evaluate the holdout.")
    parser.add_argument("--output-base", default=str(OUTPUTS))
    parser.add_argument("--dataset", default=str(DATASET_GLOSSES))
    parser.add_argument("--prompt-file", default=str(PROMPT))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="If > 0, replaces --epochs.")
    parser.add_argument("--log-dir", default=str(LOGS))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Catalog ids. Default: all.")
    return parser.parse_args()


def log_line(log_fp, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


def prepare_examples(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw.get("dataset", raw.get("examples", []))
    return [
        {
            "glosses": unify_fingerspelled_glosses(item["glosses"]),
            "spanish": item["spanish"],
        }
        for item in items
    ]


def export_gguf(model, tokenizer, output_dir: Path, log_fp) -> Path:
    for folder in (TEMP_GGUF, TEMP_GGUF_EXTRA):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    TEMP_GGUF.mkdir(parents=True, exist_ok=True)

    log_line(log_fp, "Merging weights and exporting GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(str(TEMP_GGUF), tokenizer, quantization_method="q4_k_m")

    destination = output_dir.parent / f"{output_dir.name}_gguf"
    destination.mkdir(parents=True, exist_ok=True)
    for source in list(TEMP_GGUF.glob("*.gguf")) + list(TEMP_GGUF_EXTRA.glob("*.gguf")):
        final = destination / source.name
        if final.exists():
            final.unlink()
        shutil.move(str(source), str(final))
    for folder in (TEMP_GGUF, TEMP_GGUF_EXTRA):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    return destination


def evaluate_holdout(gguf_dir: Path, test_raw, references: dict, system_prompt: str, log_fp) -> dict:
    candidates = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(f"No GGUF found in {gguf_dir}")
    gguf = candidates[0]
    log_line(log_fp, f"Loading GGUF engine for direct evaluation: {gguf.name}")
    llm = load_gguf(gguf)

    predictions = []
    reference_lists = []
    expected = []
    sources = []
    latencies = []
    strict_hits = normalized_hits = 0
    try:
        for item in test_raw:
            glosses = gloss_text(item["glosses"])
            target = item["spanish"].strip()
            options = references[glosses]
            prediction, latency = translate_gguf(llm, system_prompt, glosses)
            predictions.append(prediction)
            reference_lists.append(options)
            expected.append(target)
            sources.append(glosses)
            latencies.append(latency)
            strict, normalized = matches(prediction, options)
            strict_hits += int(strict)
            normalized_hits += int(normalized)
    finally:
        del llm
        gc.collect()

    total = len(test_raw)
    scores = score_all(predictions, reference_lists, sources)
    return {
        "total_test_samples": total,
        "accuracy_strict_percent": round(100.0 * strict_hits / total, 2),
        "accuracy_normalized_percent": round(100.0 * normalized_hits / total, 2),
        "bleu_score": scores["bleu_score"],
        "rouge_l_score": scores["rouge_l_score"],
        "meteor_score": scores["meteor_score"],
        "bertscore_f1": scores["bertscore_f1"],
        "comet_score": scores["comet_score"],
        "bleurt_score": scores["bleurt_score"],
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "evaluated_examples": [
            {"glosses": source, "expected": target, "predicted": prediction}
            for source, target, prediction in zip(sources, expected, predictions)
        ],
    }


def train_one_model(model_cfg: dict, output_dir: Path, dataset_path: Path, system_prompt: str, epochs: int, max_steps: int, log_fp) -> dict:
    name = model_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_line(log_fp, f"Loading {name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg.get("chat_template", "chatml"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    examples = prepare_examples(dataset_path)
    dataset = Dataset.from_list(examples).train_test_split(test_size=0.15, seed=3407)
    train_raw, test_raw = dataset["train"], dataset["test"]

    def format_prompts(batch):
        texts = []
        for glosses, spanish in zip(batch["glosses"], batch["spanish"]):
            messages = gloss_messages(system_prompt, gloss_text(glosses))
            messages.append({"role": "assistant", "content": spanish})
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_dataset = train_raw.map(format_prompts, batched=True)
    large = "3b" in name.lower()
    args = {
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
        "max_length": 512,
        "packing": False,
        "report_to": "none",
    }
    if max_steps > 0:
        args["max_steps"] = max_steps
    else:
        args["num_train_epochs"] = epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**args),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=model_cfg.get("instruction_part", "<|im_start|>user\n"),
        response_part=model_cfg.get("response_part", "<|im_start|>assistant\n"),
    )

    log_line(log_fp, f"Starting training for {name}...")
    result = trainer.train()
    loss = result.training_loss if hasattr(result, "training_loss") else None
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    gguf_dir = export_gguf(model, tokenizer, output_dir, log_fp)

    del model, tokenizer, trainer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    references = defaultdict(list)
    for item in examples:
        key = gloss_text(item["glosses"])
        text = item["spanish"].strip()
        if text not in references[key]:
            references[key].append(text)

    log_line(log_fp, "Starting multi-reference evaluation on the GGUF binary...")
    metrics = evaluate_holdout(gguf_dir, test_raw, references, system_prompt, log_fp)
    metrics.update({"model_name": name, "format": "GGUF_Q4_K_M", "train_loss_final": loss})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=4, ensure_ascii=False), encoding="utf-8")
    return metrics


def metrics_summary(metrics: dict) -> dict:
    keys = (
        "accuracy_strict_percent",
        "accuracy_normalized_percent",
        "bleu_score",
        "rouge_l_score",
        "meteor_score",
        "bertscore_f1",
        "comet_score",
        "bleurt_score",
        "avg_latency_ms",
        "train_loss_final",
    )
    return {key: metrics[key] for key in keys}


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    prompt_path = Path(args.prompt_file)
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        print(f"Error: system prompt not found at {prompt_path}", file=sys.stderr)
        return 1

    chosen, missing = select_models(args.models)
    if missing:
        print(f"Unknown ids: {', '.join(missing)}", file=sys.stderr)
        return 1

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{stamp}.log"
    summary_path = output_base / "training_summary.json"
    summary = load_summary(summary_path)
    batch = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
        "models_requested": [model["id"] for model in chosen],
        "results": [],
    }
    failures = 0

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Starting batch of {len(chosen)} models with GGUF evaluation")
        log_line(log_fp, f"Output base: {output_base}")
        for index, model in enumerate(chosen, start=1):
            log_line(log_fp, f"\n[{index}/{len(chosen)}] Training {model['label']} ({model['name']})")
            started = datetime.now(timezone.utc).isoformat()
            output_dir = output_base / model_folder(model["name"])
            try:
                metrics = train_one_model(
                    model, output_dir, dataset_path, system_prompt, args.epochs, args.max_steps, log_fp
                )
            except Exception as exc:
                failures += 1
                log_line(log_fp, f"FAIL {model['id']}: {exc}")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if not args.continue_on_error:
                    log_line(log_fp, "Aborting batch because of an error.")
                    break
                continue

            result = {
                "id": model["id"],
                "model_name": model["name"],
                "label": model["label"],
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "output_dir": str(output_dir),
                "metrics_file": str(output_dir / "metrics.json"),
                "metrics": metrics_summary(metrics),
            }
            batch["results"].append(result)
            summary["runs"].append(result)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            scored = result["metrics"]
            log_line(
                log_fp,
                f"OK {model['id']} (GGUF) | acc_norm={scored['accuracy_normalized_percent']}% "
                f"bleu={scored['bleu_score']} rouge={scored['rouge_l_score']} meteor={scored['meteor_score']} "
                f"bert={scored['bertscore_f1']} comet={scored['comet_score']} bleurt={scored['bleurt_score']} "
                f"latency_ms={scored['avg_latency_ms']}",
            )

        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        batch["failures"] = failures
        batch_path = output_base / f"training_batch_{stamp}.json"
        batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        log_line(log_fp, f"\nBatch summary: {batch_path}")
        log_line(log_fp, f"Cumulative summary: {summary_path}")
        log_line(log_fp, f"Done. Failures: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
