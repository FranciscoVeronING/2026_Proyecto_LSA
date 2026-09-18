"""Evalúa los GGUF sobre dataset_trial y arma la comparativa en Markdown."""
from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_OUTPUT_BASE,
    DEFAULT_TRIAL_DATASET,
    MODELS,
    model_folder_name,
)
from dataset import load_trial_groups, load_trial_samples
from gguf_export import find_q4_gguf
from io_utils import log_line, save_json
from metrics import evaluar_pares, load_gguf_engine
from prompts import load_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa GGUF entrenados sobre el trial set (español → glosas).",
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--dataset", default=str(DEFAULT_TRIAL_DATASET))
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def _gguf_dir_for(output_base: Path, model_name: str) -> Path:
    folder = model_folder_name(model_name)
    return output_base / f"{folder}_gguf"


def rank_key(metrics: dict) -> tuple:
    return (
        metrics["accuracy_normalized_percent"],
        metrics["bleu_score"],
        metrics["rouge_l_score"],
        metrics["meteor_score"],
        -(metrics["avg_latency_ms"] or 10**9),
    )


def accuracy_por_categoria(ejemplos: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for item in ejemplos:
        category = item.get("category") or "(sin categoría)"
        buckets[category].append(bool(item.get("exact_normalized")))
    out: dict[str, dict[str, float]] = {}
    for category, flags in buckets.items():
        total = len(flags)
        hits = sum(flags)
        out[category] = {
            "n": total,
            "accuracy_normalized_percent": round((hits / total) * 100, 2) if total else 0.0,
        }
    return out


def _pred_por_espanol(ejemplos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["espanol"]: item for item in ejemplos}


def build_per_example(groups: list[dict[str, Any]], results: list[dict]) -> list[dict]:
    """Una categoría del trial, con la primera paráfrasis como fila de la tabla."""
    rows = []
    lookups = [(m["id"], m["label"], _pred_por_espanol(m["ejemplos_evaluados"])) for m in results]
    for group in groups:
        representative = group["spanish"][0]
        predictions = []
        for model_id, label, by_spanish in lookups:
            ejemplo = by_spanish.get(representative)
            if ejemplo is None:
                continue
            predictions.append(
                {
                    "id": model_id,
                    "label": label,
                    "predicho": ejemplo["predicho"],
                    "exact_normalized": ejemplo["exact_normalized"],
                }
            )
        glosas_mostradas = (
            " ".join(group["glosses"]) if isinstance(group["glosses"], list) else str(group["glosses"])
        )
        rows.append(
            {
                "category": group["category"],
                "espanol": representative,
                "variantes": group["spanish"],
                "glosas": glosas_mostradas,
                "predictions": predictions,
            }
        )
    return rows


def fmt(value, width: int = 8) -> str:
    if value is None:
        return str("-").rjust(width)
    if isinstance(value, float):
        return f"{value:.2f}".rjust(width)
    return str(value).rjust(width)


def print_table(results: list[dict]) -> None:
    print()
    print("=" * 108)
    print("COMPARATIVA dataset_trial  (español → glosas)")
    print("=" * 108)
    print(
        f"{'#':>2}  {'modelo':<28} {'back':<10} {'acc_n':>7} {'acc_s':>7} "
        f"{'bleu':>7} {'rouge':>7} {'meteor':>7} {'lat_ms':>8}"
    )
    print("-" * 108)
    for i, model in enumerate(results, start=1):
        print(
            f"{i:>2}  {model['id']:<28} {model['backend']:<10} "
            f"{fmt(model['accuracy_normalized_percent'], 7)} {fmt(model['accuracy_strict_percent'], 7)} "
            f"{fmt(model['bleu_score'], 7)} {fmt(model['rouge_l_score'], 7)} {fmt(model['meteor_score'], 7)} "
            f"{fmt(model.get('avg_latency_ms'), 8)}"
        )
    print("=" * 108)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_markdown(report: dict) -> str:
    ranked = report["ranking"]
    lines = [
        "# Comparativa de modelos — dataset_trial",
        "",
        "- Tarea: español → glosas LSA",
        f"- Dataset: `{report['dataset']}`",
        f"- Ejemplos: {report['n_samples']}",
        f"- Fecha: {report['evaluated_at']}",
        "",
        "## Ranking",
        "",
        "| # | Modelo | Backend | Acc. norm. % | Acc. estricta % | BLEU | ROUGE-L | METEOR | Latencia ms |",
        "|---|--------|---------|--------------|-----------------|------|---------|--------|-------------|",
    ]
    for i, model in enumerate(ranked, start=1):
        lat = model.get("avg_latency_ms")
        lat_s = f"{lat:.2f}" if isinstance(lat, (int, float)) else "-"
        lines.append(
            f"| {i} | {model['label']} | {model['backend']} | "
            f"{model['accuracy_normalized_percent']:.2f} | {model['accuracy_strict_percent']:.2f} | "
            f"{model['bleu_score']:.2f} | {model['rouge_l_score']:.2f} | "
            f"{model['meteor_score']:.2f} | {lat_s} |"
        )

    if report.get("failures"):
        lines += ["", "## Fallos", ""]
        for fail in report["failures"]:
            lines.append(f"- `{fail['id']}`: {fail['error']}")

    lines += ["", "## Predicciones por ejemplo", ""]
    for item in report.get("per_example", []):
        lines.append(f"### {item['category']}")
        lines.append("")
        lines.append(f"- Español (variante mostrada): `{_cell(item['espanol'])}`")
        if len(item.get("variantes", [])) > 1:
            lines.append("- Otras variantes:")
            for variante in item["variantes"][1:]:
                lines.append(f"  - {_cell(variante)}")
        lines.append(f"- Glosas esperadas: `{_cell(item['glosas'])}`")
        lines.append("")
        lines.append("| Modelo | Predicción | Exacta (norm.) |")
        lines.append("|--------|------------|----------------|")
        for pred in item["predictions"]:
            mark = "sí" if pred["exact_normalized"] else "no"
            lines.append(f"| {pred['label']} | {_cell(pred['predicho'])} | {mark} |")
        lines.append("")

    lines += ["", "## Métricas por categoría", ""]
    categories = sorted({cat for model in ranked for cat in model.get("by_category", {})})
    if categories and ranked:
        header = "| Categoría | " + " | ".join(model["id"] for model in ranked) + " |"
        sep = "|-----------|" + "|".join(["--------"] * len(ranked)) + "|"
        lines += [header, sep]
        for cat in categories:
            cells = []
            for model in ranked:
                cat_m = model.get("by_category", {}).get(cat)
                cells.append(f"{cat_m['accuracy_normalized_percent']:.0f}%" if cat_m else "-")
            lines.append(f"| `{cat}` | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: no está el trial set en {dataset_path}", file=sys.stderr)
        return 1

    selected = MODELS
    if args.models:
        wanted = set(args.models)
        selected = [model for model in MODELS if model["id"] in wanted]
        missing = wanted - {model["id"] for model in selected}
        if missing:
            print(f"IDs desconocidos: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    system_prompt = load_system_prompt()
    samples = load_trial_samples(dataset_path)
    groups = load_trial_groups(dataset_path)

    refs: dict[str, list[str]] = {}
    for sample in samples:
        refs.setdefault(sample["spanish"], [])
        if sample["glosses"] not in refs[sample["spanish"]]:
            refs[sample["spanish"]].append(sample["glosses"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / "eval_trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"comparativa_trial_{stamp}.log"

    results: list[dict] = []
    failures: list[dict] = []

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Trial eval: {len(samples)} oraciones, {len(selected)} modelos")
        for model_cfg in selected:
            gguf_dir = _gguf_dir_for(output_base, model_cfg["name"])
            try:
                gguf_file = find_q4_gguf(gguf_dir)
            except FileNotFoundError as exc:
                failures.append({"id": model_cfg["id"], "error": str(exc)})
                log_line(log_fp, f"SKIP {model_cfg['id']}: {exc}")
                continue

            log_line(log_fp, f"Evaluando {model_cfg['id']} ({gguf_file.name})")
            try:
                llm = load_gguf_engine(gguf_file)
                metrics = evaluar_pares(
                    llm=llm,
                    system_prompt=system_prompt,
                    test_items=samples,
                    refs_por_espanol=refs,
                    model_name=model_cfg["name"],
                )
            except Exception as exc:
                failures.append({"id": model_cfg["id"], "error": str(exc)})
                log_line(log_fp, f"FAIL {model_cfg['id']}: {exc}")
                gc.collect()
                continue
            del llm
            gc.collect()

            row = {
                "id": model_cfg["id"],
                "label": model_cfg["label"],
                "backend": "gguf",
                "gguf": str(gguf_file),
                "accuracy_strict_percent": metrics["accuracy_strict_percent"],
                "accuracy_normalized_percent": metrics["accuracy_normalized_percent"],
                "bleu_score": metrics["bleu_score"],
                "rouge_l_score": metrics["rouge_l_score"],
                "meteor_score": metrics["meteor_score"],
                "avg_latency_ms": metrics["avg_latency_ms"],
                "by_category": accuracy_por_categoria(metrics["ejemplos_evaluados"]),
                "ejemplos_evaluados": metrics["ejemplos_evaluados"],
            }
            results.append(row)
            log_line(
                log_fp,
                f"OK {model_cfg['id']} acc_norm={metrics['accuracy_normalized_percent']}% "
                f"bleu={metrics['bleu_score']}",
            )

        results.sort(key=rank_key, reverse=True)
        report = {
            "task": "spanish_to_lsa_glosses",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "n_samples": len(samples),
            "ranking": [{k: v for k, v in m.items() if k != "ejemplos_evaluados"} for m in results],
            "per_example": build_per_example(groups, results) if results else [],
            "failures": failures,
        }

        json_path = out_dir / f"comparativa_trial_{stamp}.json"
        md_path = out_dir / f"comparativa_trial_{stamp}.md"
        save_json(json_path, report)
        md_text = build_markdown(report)
        md_path.write_text(md_text, encoding="utf-8")
        save_json(out_dir / "comparativa_trial_latest.json", report)
        (out_dir / "comparativa_trial_latest.md").write_text(md_text, encoding="utf-8")
        log_line(log_fp, f"Escrito {md_path}")

    if results:
        print_table(results)
        print(f"Mejor modelo: {results[0]['label']} ({results[0]['id']})")
    print(f"Reporte Markdown: {out_dir / f'comparativa_trial_{stamp}.md'}")
    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())