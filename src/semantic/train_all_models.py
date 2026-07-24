"""Entrena secuencialmente los 5 LLM candidatos y guarda metricas por carpeta."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from train_runner import DEFAULT_OUTPUT_BASE, model_folder_name, train_single_model

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"

MODELS = [
    {
        "id": "qwen2.5-0.5b",
        "name": "unsloth/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B (ultra liviano)",
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "unsloth/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B",
    },
    {
        "id": "qwen2.5-3b",
        "name": "unsloth/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B (mayor capacidad)",
    },
    {
        "id": "llama-3.2-1b",
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "label": "Llama 3.2 1B (Meta)",
    },
    {
        "id": "gemma-2-2b",
        "name": "unsloth/gemma-2-2b-it",
        "label": "Gemma 2 2B (Google)",
    },
    {
        "id": "phi-3-mini-4k",
        "name": "unsloth/Phi-3-mini-4k-instruct",
        "label": "Phi-3 mini 4k (Microsoft)",
    },
    {
        "id": "smollm2-1.7b",
        "name": "unsloth/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B",
    },
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena los 5 LLM en secuencia, uno tras otro.",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Carpeta base de salida (cada modelo tiene subcarpeta).",
    )
    parser.add_argument(
        "--dataset",
        default=str(SCRIPT_DIR / "dataset_glosas.json"),
        help="Dataset JSON para entrenamiento.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Pasos de entrenamiento por modelo.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Carpeta donde se guarda el log de la corrida.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Si un modelo falla, continuar con el siguiente.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="IDs opcionales a entrenar (ej: qwen2.5-0.5b phi-3-mini-4k). Default: los 5.",
    )
    return parser.parse_args()


def log_line(log_fp, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_fp.write(line + "\n")
    log_fp.flush()


def load_summary(summary_path: Path) -> dict:
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"runs": []}


def save_summary(summary_path: Path, summary: dict) -> None:
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    selected = MODELS
    if args.models:
        wanted = set(args.models)
        selected = [m for m in MODELS if m["id"] in wanted]
        missing = wanted - {m["id"] for m in selected}
        if missing:
            print(f"IDs desconocidos: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_all_{run_stamp}.log"
    summary_path = output_base / "training_summary.json"

    summary = load_summary(summary_path)
    batch = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
        "models_requested": [m["id"] for m in selected],
        "results": [],
    }

    failures = 0
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Inicio batch de {len(selected)} modelos")
        log_line(log_fp, f"Salida base: {output_base}")

        for index, model_cfg in enumerate(selected, start=1):
            log_line(log_fp, f"[{index}/{len(selected)}] Entrenando {model_cfg['label']} ({model_cfg['name']})")
            started = datetime.now(timezone.utc).isoformat()

            try:
                metrics = train_single_model(
                    model_name=model_cfg["name"],
                    output_base=output_base,
                    dataset_path=dataset_path,
                    max_steps=args.max_steps,
                )
                exit_code = 0
            except Exception as exc:
                failures += 1
                exit_code = 1
                metrics = None
                log_line(log_fp, f"FAIL {model_cfg['id']}: {exc}")
                if not args.continue_on_error:
                    log_line(log_fp, "Abortando batch por error.")
                    break
                continue

            finished = datetime.now(timezone.utc).isoformat()
            model_dir = output_base / model_folder_name(model_cfg["name"])

            result = {
                "id": model_cfg["id"],
                "model_name": model_cfg["name"],
                "label": model_cfg["label"],
                "started_at": started,
                "finished_at": finished,
                "exit_code": exit_code,
                "output_dir": str(model_dir),
                "metrics_file": str(model_dir / "metrics.json"),
                "metrics": {
                    "accuracy_exact_match_percent": metrics["accuracy_exact_match_percent"],
                    "bleu_score": metrics["bleu_score"],
                    "rouge_l_score": metrics["rouge_l_score"],
                    "train_loss_final": metrics["train_loss_final"],
                },
            }
            batch["results"].append(result)
            summary["runs"].append(result)
            save_summary(summary_path, summary)

            log_line(
                log_fp,
                f"OK {model_cfg['id']} | acc={result['metrics']['accuracy_exact_match_percent']}% "
                f"bleu={result['metrics']['bleu_score']} rouge={result['metrics']['rouge_l_score']}",
            )

        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        batch["failures"] = failures
        batch_path = output_base / f"training_batch_{run_stamp}.json"
        batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        log_line(log_fp, f"Resumen batch: {batch_path}")
        log_line(log_fp, f"Resumen acumulado: {summary_path}")
        log_line(log_fp, f"Fin. Fallos: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
