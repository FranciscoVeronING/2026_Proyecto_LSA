"""Score exported GGUF models on dataset_trial.json and write the comparison."""
from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DEFAULT_OUTPUT_BASE, DEFAULT_TRIAL_DATASET, gguf_dir_for, select_models
from dataset import load_trial_groups, load_trial_samples
from gguf import find_q4_gguf, load_gguf_engine
from io_utils import log_line, save_json
from metrics import copy_scores, evaluate_pairs
from prompts import load_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained GGUF models on the trial set (Spanish to glosses).",
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--dataset", default=str(DEFAULT_TRIAL_DATASET))
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def rank_key(metrics: dict) -> tuple:
    return (
        metrics["accuracy_normalized_percent"],
        metrics["bleu_score"],
        metrics["rouge_l_score"],
        metrics["meteor_score"],
        metrics["bertscore_f1"],
        metrics["comet_score"],
        metrics["bleurt_score"],
        -(metrics["avg_latency_ms"] or 10**9),
    )


def accuracy_by_category(examples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for item in examples:
        category = item.get("category") or "(no category)"
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


def _predictions_by_spanish(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["spanish"]: item for item in examples}


def build_per_example(groups: list[dict[str, Any]], results: list[dict]) -> list[dict]:
    """One trial category. The table row uses the first paraphrase."""
    rows = []
    lookups = [(m["id"], m["label"], _predictions_by_spanish(m["evaluated_examples"])) for m in results]
    for group in groups:
        representative = group["spanish"][0]
        predictions = []
        for model_id, label, by_spanish in lookups:
            example = by_spanish.get(representative)
            if example is None:
                continue
            predictions.append(
                {
                    "id": model_id,
                    "label": label,
                    "predicted": example["predicted"],
                    "exact_normalized": example["exact_normalized"],
                }
            )
        shown_glosses = (
            " ".join(group["glosses"]) if isinstance(group["glosses"], list) else str(group["glosses"])
        )
        rows.append(
            {
                "category": group["category"],
                "spanish": representative,
                "variants": group["spanish"],
                "glosses": shown_glosses,
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
    print("=" * 132)
    print("TRIAL COMPARISON  (Spanish to glosses)")
    print("=" * 132)
    print(
        f"{'#':>2}  {'model':<28} {'back':<10} {'acc_n':>7} {'acc_s':>7} "
        f"{'bleu':>7} {'rouge':>7} {'meteor':>7} {'bert':>7} {'comet':>7} {'bleurt':>7} {'lat_ms':>8}"
    )
    print("-" * 132)
    for i, model in enumerate(results, start=1):
        print(
            f"{i:>2}  {model['id']:<28} {model['backend']:<10} "
            f"{fmt(model['accuracy_normalized_percent'], 7)} {fmt(model['accuracy_strict_percent'], 7)} "
            f"{fmt(model['bleu_score'], 7)} {fmt(model['rouge_l_score'], 7)} {fmt(model['meteor_score'], 7)} "
            f"{fmt(model['bertscore_f1'], 7)} {fmt(model['comet_score'], 7)} {fmt(model['bleurt_score'], 7)} "
            f"{fmt(model.get('avg_latency_ms'), 8)}"
        )
    print("=" * 132)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_markdown(report: dict) -> str:
    ranked = report["ranking"]
    lines = [
        "# Model comparison — dataset_trial",
        "",
        "- Task: Spanish to LSA glosses",
        f"- Dataset: `{report['dataset']}`",
        f"- Examples: {report['n_samples']}",
        f"- Date: {report['evaluated_at']}",
        "",
        "## Ranking",
        "",
        "| # | Model | Backend | Norm. acc. % | Strict acc. % | BLEU | ROUGE-L | METEOR | BERTScore | COMET | BLEURT | Latency ms |",
        "|---|--------|---------|--------------|-----------------|------|---------|--------|-----------|-------|--------|-------------|",
    ]
    for i, model in enumerate(ranked, start=1):
        lat = model.get("avg_latency_ms")
        lat_s = f"{lat:.2f}" if isinstance(lat, (int, float)) else "-"
        lines.append(
            f"| {i} | {model['label']} | {model['backend']} | "
            f"{model['accuracy_normalized_percent']:.2f} | {model['accuracy_strict_percent']:.2f} | "
            f"{model['bleu_score']:.2f} | {model['rouge_l_score']:.2f} | "
            f"{model['meteor_score']:.2f} | {model['bertscore_f1']:.2f} | {model['comet_score']:.2f} | "
            f"{model['bleurt_score']:.2f} | {lat_s} |"
        )

    if report.get("failures"):
        lines += ["", "## Failures", ""]
        for fail in report["failures"]:
            lines.append(f"- `{fail['id']}`: {fail['error']}")

    lines += ["", "## Predictions by example", ""]
    for item in report.get("per_example", []):
        lines.append(f"### {item['category']}")
        lines.append("")
        lines.append(f"- Spanish (shown variant): `{_cell(item['spanish'])}`")
        if len(item.get("variants", [])) > 1:
            lines.append("- Other variants:")
            for variant in item["variants"][1:]:
                lines.append(f"  - {_cell(variant)}")
        lines.append(f"- Expected glosses: `{_cell(item['glosses'])}`")
        lines.append("")
        lines.append("| Model | Prediction | Exact (normalized) |")
        lines.append("|-------|------------|--------------------|")
        for pred in item["predictions"]:
            mark = "yes" if pred["exact_normalized"] else "no"
            lines.append(f"| {pred['label']} | {_cell(pred['predicted'])} | {mark} |")
        lines.append("")

    lines += ["", "## Metrics by category", ""]
    categories = sorted({cat for model in ranked for cat in model.get("by_category", {})})
    if categories and ranked:
        header = "| Category | " + " | ".join(model["id"] for model in ranked) + " |"
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
        print(f"Error: trial set not found at {dataset_path}", file=sys.stderr)
        return 1

    try:
        selected = select_models(args.models)
    except ValueError as exc:
        print(exc, file=sys.stderr)
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
    log_path = out_dir / f"trial_compare_{stamp}.log"

    results: list[dict] = []
    failures: list[dict] = []

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Trial eval: {len(samples)} sentences, {len(selected)} models")
        for model_cfg in selected:
            gguf_dir = gguf_dir_for(output_base, model_cfg["name"])
            try:
                gguf_file = find_q4_gguf(gguf_dir)
            except FileNotFoundError as exc:
                failures.append({"id": model_cfg["id"], "error": str(exc)})
                log_line(log_fp, f"SKIP {model_cfg['id']}: {exc}")
                continue

            log_line(log_fp, f"Evaluating {model_cfg['id']} ({gguf_file.name})")
            try:
                llm = load_gguf_engine(gguf_file)
                metrics = evaluate_pairs(
                    llm=llm,
                    system_prompt=system_prompt,
                    test_items=samples,
                    refs_by_spanish=refs,
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
                **copy_scores(metrics),
                "by_category": accuracy_by_category(metrics["evaluated_examples"]),
                "evaluated_examples": metrics["evaluated_examples"],
            }
            results.append(row)
            log_line(
                log_fp,
                f"OK {model_cfg['id']} acc_norm={metrics['accuracy_normalized_percent']}% "
                f"bleu={metrics['bleu_score']} bert={metrics['bertscore_f1']} "
                f"comet={metrics['comet_score']} bleurt={metrics['bleurt_score']}",
            )

        results.sort(key=rank_key, reverse=True)
        report = {
            "task": "spanish_to_lsa_glosses",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset_path),
            "n_samples": len(samples),
            "ranking": [{k: v for k, v in m.items() if k != "evaluated_examples"} for m in results],
            "per_example": build_per_example(groups, results) if results else [],
            "failures": failures,
        }

        json_path = out_dir / f"trial_compare_{stamp}.json"
        md_path = out_dir / f"trial_compare_{stamp}.md"
        save_json(json_path, report)
        md_text = build_markdown(report)
        md_path.write_text(md_text, encoding="utf-8")
        save_json(out_dir / "trial_compare_latest.json", report)
        (out_dir / "trial_compare_latest.md").write_text(md_text, encoding="utf-8")
        log_line(log_fp, f"Wrote {md_path}")

    if results:
        print_table(results)
        print(f"Best model: {results[0]['label']} ({results[0]['id']})")
    print(f"Markdown report: {out_dir / f'trial_compare_{stamp}.md'}")
    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())