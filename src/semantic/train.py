"""Train every model in config.MODELS, export GGUF, and score the held-out split.

Unsloth must load before torch/trl, so train_model is the first import.
"""
from __future__ import annotations

from train_model import train_and_eval_single_model

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import DEFAULT_DATASET, DEFAULT_LOG_DIR, DEFAULT_OUTPUT_BASE, model_folder_name, select_models
from io_utils import load_summary, log_line, save_json
from metrics import copy_scores
from prompts import load_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential fine-tuning: Spanish sentence to LSA glosses. Exports GGUF and evaluates.",
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--prompt-file", default=None, help="System prompt. Default: prompts/sys_prompt.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="If > 0, overrides --epochs.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model ids to train (for example qwen2.5-0.5b llama-3.2-1b). Default: all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    prompt_path = Path(args.prompt_file) if args.prompt_file else None

    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Error: dataset not found at {dataset_path}", file=sys.stderr)
        return 1

    system_prompt = load_system_prompt(prompt_path=prompt_path)

    try:
        selected = select_models(args.models)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_all_{run_stamp}.log"
    summary_path = output_base / "training_summary.json"
    summary = load_summary(summary_path)
    batch: dict = {
        "task": "spanish_to_lsa_glosses",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
        "models_requested": [model["id"] for model in selected],
        "results": [],
    }

    failures = 0
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Batch of {len(selected)} models | Spanish to LSA glosses")
        log_line(log_fp, f"Dataset: {dataset_path}")
        log_line(log_fp, f"Output: {output_base}")

        for index, model_cfg in enumerate(selected, start=1):
            log_line(log_fp, f"\n[{index}/{len(selected)}] {model_cfg['label']} ({model_cfg['name']})")
            started = datetime.now(timezone.utc).isoformat()
            model_dir = output_base / model_folder_name(model_cfg["name"])
            try:
                metrics = train_and_eval_single_model(
                    model_cfg=model_cfg,
                    output_dir=model_dir,
                    dataset_path=dataset_path,
                    system_prompt=system_prompt,
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    log_fp=log_fp,
                )
            except Exception as exc:
                failures += 1
                log_line(log_fp, f"FAIL {model_cfg['id']}: {exc}")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if not args.continue_on_error:
                    log_line(log_fp, "Aborting the batch because of an error.")
                    break
                continue

            result = {
                "id": model_cfg["id"],
                "model_name": model_cfg["name"],
                "label": model_cfg["label"],
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "output_dir": str(model_dir),
                "metrics_file": str(model_dir / "metrics.json"),
                "metrics": {**copy_scores(metrics), "train_loss_final": metrics["train_loss_final"]},
            }
            batch["results"].append(result)
            summary["runs"].append(result)
            save_json(summary_path, summary)
            log_line(
                log_fp,
                f"OK {model_cfg['id']} | acc_norm={result['metrics']['accuracy_normalized_percent']}% "
                f"bleu={result['metrics']['bleu_score']} rouge={result['metrics']['rouge_l_score']} "
                f"meteor={result['metrics']['meteor_score']} bert={result['metrics']['bertscore_f1']} "
                f"comet={result['metrics']['comet_score']} bleurt={result['metrics']['bleurt_score']} "
                f"lat={result['metrics']['avg_latency_ms']}ms",
            )

        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        batch["failures"] = failures
        batch_path = output_base / f"training_batch_{run_stamp}.json"
        save_json(batch_path, batch)
        log_line(log_fp, f"\nBatch summary: {batch_path}")
        log_line(log_fp, f"Cumulative summary: {summary_path}")
        log_line(log_fp, f"Done. Failures: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
