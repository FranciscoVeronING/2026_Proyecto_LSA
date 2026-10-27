"""Compare already trained models on dataset_trial and save the report."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from glosses import clean_output, gloss_text, matches, unify_fingerspelled_glosses
from inference import gloss_messages, load_gguf, translate_gguf
from metrics import score_all
from models import (
    DATASET_TRIAL,
    EVAL_TRIAL,
    MAX_NEW_TOKENS,
    OUTPUTS,
    PROMPT,
    model_folder,
    select_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the catalog on dataset_trial.")
    parser.add_argument("--output-base", default=str(OUTPUTS))
    parser.add_argument("--dataset", default=str(DATASET_TRIAL))
    parser.add_argument("--prompt-file", default=str(PROMPT))
    parser.add_argument("--report-dir", default=None, help="Default: <output-base>/eval_trial.")
    parser.add_argument("--models", nargs="*", default=None, help="Catalog ids. Default: all.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def load_trial(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw.get("test_dataset", raw.get("dataset", raw.get("examples", [])))
    if not isinstance(items, list) or not items:
        raise ValueError(f"No examples found in {dataset_path}")

    samples = []
    for item in items:
        glosses = unify_fingerspelled_glosses(item["glosses"])
        spanish = item["spanish"]
        if isinstance(spanish, str):
            references = [spanish.strip()]
        else:
            references = [text.strip() for text in spanish if str(text).strip()]
        if not references:
            continue
        samples.append(
            {
                "category": item.get("category", "uncategorized"),
                "glosses": glosses,
                "gloss_text": gloss_text(glosses),
                "references": references,
            }
        )
    return samples


def resolve_checkpoint(output_base: Path, model: dict) -> dict:
    folder = model_folder(model["name"])
    gguf_dir = output_base / f"{folder}_gguf"
    if gguf_dir.is_dir():
        candidates = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
        if candidates:
            return {"backend": "gguf", "path": candidates[0]}
    hf_dir = output_base / folder
    if (hf_dir / "config.json").exists():
        return {"backend": "causal_hf", "path": hf_dir}
    raise FileNotFoundError(f"No GGUF or HF checkpoint for {model['id']} ({folder})")


def predict_gguf(path: Path, system_prompt: str, samples: list[dict]) -> tuple[list[str], list[float]]:
    llm = load_gguf(path)
    predictions, latencies = [], []
    try:
        for sample in samples:
            text, latency = translate_gguf(llm, system_prompt, sample["gloss_text"])
            predictions.append(text)
            latencies.append(latency)
    finally:
        del llm
        gc.collect()
    return predictions, latencies


def predict_hf(path: Path, system_prompt: str, samples: list[dict]) -> tuple[list[str], list[float]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions, latencies = [], []
    try:
        for sample in samples:
            encoded = tokenizer.apply_chat_template(
                gloss_messages(system_prompt, sample["gloss_text"]),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            started = time.time()
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latencies.append((time.time() - started) * 1000)
            prompt_length = encoded["input_ids"].shape[1]
            text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
            predictions.append(clean_output(text.split("\n")[0]))
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, latencies


def measure(predictions: list[str], samples: list[dict], latencies: list[float]) -> dict:
    references = [sample["references"] for sample in samples]
    scores = score_all(predictions, references, [sample["gloss_text"] for sample in samples])
    examples = []
    strict_hits = normalized_hits = 0
    for pred, sample, rouge, meteor, bert, comet, bleurt in zip(
        predictions,
        samples,
        scores["rouge_l"],
        scores["meteor"],
        scores["bertscore_raw"],
        scores["comet_raw"],
        scores["bleurt_raw"],
    ):
        strict, normalized = matches(pred, sample["references"])
        strict_hits += int(strict)
        normalized_hits += int(normalized)
        examples.append(
            {
                "category": sample["category"],
                "glosses": sample["gloss_text"],
                "references": sample["references"],
                "predicted": pred,
                "exact_strict": strict,
                "exact_normalized": normalized,
                "rouge_l": rouge,
                "meteor": meteor,
                "bertscore_f1": bert,
                "comet": comet,
                "bleurt": bleurt,
            }
        )

    by_category: dict[str, dict] = {}
    for row in examples:
        bucket = by_category.setdefault(row["category"], {"n": 0, "exact_normalized": 0, "rouge_l": 0.0})
        bucket["n"] += 1
        bucket["exact_normalized"] += int(row["exact_normalized"])
        bucket["rouge_l"] += row["rouge_l"]

    total = len(samples)
    return {
        "total_samples": total,
        "accuracy_strict_percent": round(100.0 * strict_hits / total, 2),
        "accuracy_normalized_percent": round(100.0 * normalized_hits / total, 2),
        "bleu_score": scores["bleu_score"],
        "rouge_l_score": scores["rouge_l_score"],
        "meteor_score": scores["meteor_score"],
        "bertscore_f1": scores["bertscore_f1"],
        "comet_score": scores["comet_score"],
        "bleurt_score": scores["bleurt_score"],
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "by_category": {
            category: {
                "n": data["n"],
                "accuracy_normalized_percent": round(100.0 * data["exact_normalized"] / data["n"], 2),
                "rouge_l_score": round(100.0 * data["rouge_l"] / data["n"], 2),
            }
            for category, data in by_category.items()
        },
        "examples": examples,
    }


def evaluate_model(model: dict, backend: dict, system_prompt: str, samples: list[dict]) -> dict:
    if backend["backend"] == "gguf":
        predictions, latencies = predict_gguf(backend["path"], system_prompt, samples)
    elif backend["backend"] == "causal_hf":
        predictions, latencies = predict_hf(backend["path"], system_prompt, samples)
    else:
        raise ValueError(f"Unknown backend: {backend['backend']}")
    metrics = measure(predictions, samples, latencies)
    metrics.update(
        {
            "id": model["id"],
            "model_name": model["name"],
            "label": model["label"],
            "backend": backend["backend"],
            "checkpoint": str(backend["path"]),
        }
    )
    return metrics


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


def build_examples(samples: list[dict], results: list[dict]) -> list[dict]:
    rows = []
    for index, sample in enumerate(samples):
        predictions = []
        for metrics in results:
            example = metrics["examples"][index]
            predictions.append(
                {
                    "id": metrics["id"],
                    "label": metrics["label"],
                    "predicted": example["predicted"],
                    "exact_normalized": example["exact_normalized"],
                }
            )
        rows.append(
            {
                "category": sample["category"],
                "glosses": sample["gloss_text"],
                "references": sample["references"],
                "predictions": predictions,
            }
        )
    return rows


def markdown(report: dict) -> str:
    ranking = report["ranking"]
    lines = [
        "# Model comparison — dataset_trial",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Examples: {report['n_samples']}",
        f"- Date: {report['evaluated_at']}",
        "",
        "## Ranking",
        "",
        "| # | Model | Backend | Norm. acc. % | Strict acc. % | BLEU | ROUGE-L | METEOR | BERTScore | COMET | BLEURT | Latency ms |",
        "|---|-------|---------|--------------|---------------|------|---------|--------|-----------|-------|--------|------------|",
    ]
    for index, row in enumerate(ranking, start=1):
        latency = row.get("avg_latency_ms")
        lines.append(
            f"| {index} | {row['label']} | {row['backend']} | {row['accuracy_normalized_percent']:.2f} | "
            f"{row['accuracy_strict_percent']:.2f} | {row['bleu_score']:.2f} | {row['rouge_l_score']:.2f} | "
            f"{row['meteor_score']:.2f} | {row['bertscore_f1']:.2f} | {row['comet_score']:.2f} | "
            f"{row['bleurt_score']:.2f} | {latency if latency is not None else '-'} |"
        )
    if report.get("failures"):
        lines += ["", "## Failures", ""]
        lines += [f"- `{failure['id']}`: {failure['error']}" for failure in report["failures"]]

    lines += ["", "## Predictions by example", ""]
    for item in report.get("per_example", []):
        lines += [f"### {item['category']}", "", f"- Glosses: `{item['glosses']}`", "- References:"]
        lines += [f"  - {ref}" for ref in item["references"]]
        lines += ["", "| Model | Prediction | Exact (norm.) |", "|-------|------------|---------------|"]
        for pred in item["predictions"]:
            mark = "yes" if pred["exact_normalized"] else "no"
            text = pred["predicted"].replace("|", "\\|")
            lines.append(f"| {pred['label']} | {text} | {mark} |")
        lines.append("")

    categories = sorted({cat for row in ranking for cat in row.get("by_category", {})})
    if categories:
        lines += ["", "## Metrics by category", ""]
        lines.append("| Category | " + " | ".join(row["id"] for row in ranking) + " |")
        lines.append("|----------|" + "|".join(["--------"] * len(ranking)) + "|")
        for category in categories:
            cells = []
            for row in ranking:
                data = row.get("by_category", {}).get(category)
                cells.append(f"{data['accuracy_normalized_percent']:.0f}%" if data else "-")
            lines.append(f"| `{category}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def print_table(results: list[dict]) -> None:
    def cell(value, width=7) -> str:
        if value is None:
            return "-".rjust(width)
        if isinstance(value, float):
            return f"{value:.2f}".rjust(width)
        return str(value).rjust(width)

    print("\n" + "=" * 132)
    print("COMPARISON dataset_trial")
    print("=" * 132)
    print(
        f"{'#':>2}  {'model':<28} {'back':<10} {'acc_n':>7} {'acc_s':>7} "
        f"{'bleu':>7} {'rouge':>7} {'meteor':>7} {'bert':>7} {'comet':>7} {'bleurt':>7} {'lat_ms':>8}"
    )
    print("-" * 132)
    for index, row in enumerate(results, start=1):
        print(
            f"{index:>2}  {row['id']:<28} {row['backend']:<10} "
            f"{cell(row['accuracy_normalized_percent'])} {cell(row['accuracy_strict_percent'])} "
            f"{cell(row['bleu_score'])} {cell(row['rouge_l_score'])} {cell(row['meteor_score'])} "
            f"{cell(row['bertscore_f1'])} {cell(row['comet_score'])} {cell(row['bleurt_score'])} "
            f"{cell(row.get('avg_latency_ms'), 8)}"
        )
    print("=" * 132)


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt_file)
    report_dir = Path(args.report_dir) if args.report_dir else output_base / EVAL_TRIAL.name
    if not dataset_path.exists():
        print(f"Error: dataset does not exist: {dataset_path}", file=sys.stderr)
        return 1
    if not prompt_path.exists():
        print(f"Error: system prompt does not exist: {prompt_path}", file=sys.stderr)
        return 1

    chosen, missing = select_models(args.models)
    if missing:
        print(f"Unknown ids: {', '.join(missing)}", file=sys.stderr)
        return 1

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    samples = load_trial(dataset_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    print(f"Dataset: {dataset_path} ({len(samples)} examples)")
    print(f"Candidate models: {', '.join(model['id'] for model in chosen)}")

    for index, model in enumerate(chosen, start=1):
        print(f"\n[{index}/{len(chosen)}] Evaluating {model['label']} ({model['id']})")
        try:
            backend = resolve_checkpoint(output_base, model)
            print(f"  backend={backend['backend']}  path={backend['path']}")
            metrics = evaluate_model(model, backend, system_prompt, samples)
            results.append(metrics)
            print(
                f"  acc_norm={metrics['accuracy_normalized_percent']}%  bleu={metrics['bleu_score']}  "
                f"rouge={metrics['rouge_l_score']}  meteor={metrics['meteor_score']}  "
                f"bert={metrics['bertscore_f1']}  comet={metrics['comet_score']}  "
                f"bleurt={metrics['bleurt_score']}  lat={metrics['avg_latency_ms']}ms"
            )
        except Exception as exc:
            failures.append({"id": model["id"], "error": str(exc)})
            print(f"  FAIL {model['id']}: {exc}")
            gc.collect()
            if args.stop_on_error:
                break

    results.sort(key=rank_key, reverse=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "n_samples": len(samples),
        "ranking": [{key: value for key, value in row.items() if key != "examples"} for row in results],
        "per_example": build_examples(samples, results) if results else [],
        "failures": failures,
        "details": results,
    }
    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    md_text = markdown(report)
    json_path = report_dir / f"trial_comparison_{stamp}.json"
    md_path = report_dir / f"trial_comparison_{stamp}.md"
    for path, text in (
        (json_path, json_text),
        (md_path, md_text),
        (report_dir / "trial_comparison_latest.json", json_text),
        (report_dir / "trial_comparison_latest.md", md_text),
    ):
        path.write_text(text, encoding="utf-8")

    if results:
        print_table(results)
        print(f"Best model: {results[0]['label']} ({results[0]['id']})")
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if failures:
        print(f"Failures: {len(failures)}")
        return 1
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
