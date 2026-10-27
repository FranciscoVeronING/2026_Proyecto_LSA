"""Score gloss predictions. This module is not named evaluate: Hugging Face owns that name.

Lexical: accuracy, BLEU, ROUGE-L, METEOR.
Neural: BERTScore, COMET, and BLEURT, in metrics_neural.py.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gguf import predict_glosses
from gloss import normalize_for_metrics
from metrics_neural import mean_percent, score_neural_metrics

SCORE_KEYS = (
    "accuracy_strict_percent",
    "accuracy_normalized_percent",
    "bleu_score",
    "rouge_l_score",
    "meteor_score",
    "bertscore_f1",
    "comet_score",
    "bleurt_score",
    "avg_latency_ms",
)


def _max_score(metric, pred: str, refs: Sequence[str], key: str) -> float:
    return max(metric.compute(predictions=[pred], references=[ref])[key] for ref in refs)


def evaluate_pairs(
    llm: Any,
    system_prompt: str,
    test_items: Sequence[dict[str, str]],
    refs_by_spanish: dict[str, list[str]],
    model_name: str,
) -> dict[str, Any]:
    import evaluate as hf_evaluate

    bleu_metric = hf_evaluate.load("bleu")
    rouge_metric = hf_evaluate.load("rouge")
    meteor_metric = hf_evaluate.load("meteor")

    predictions: list[str] = []
    refs_multi: list[list[str]] = []
    sources: list[str] = []
    latencies_ms: list[float] = []
    examples: list[dict[str, Any]] = []
    exact_strict = 0
    exact_norm = 0

    for item in test_items:
        spanish = item["spanish"]
        gold = item["glosses"]
        options = refs_by_spanish.get(spanish, [gold])
        predicted, latency_ms = predict_glosses(llm, system_prompt, spanish)
        latencies_ms.append(latency_ms)
        exact_ok = predicted in options
        norm_ok = any(
            normalize_for_metrics(predicted) == normalize_for_metrics(ref) for ref in options
        )
        if exact_ok:
            exact_strict += 1
        if norm_ok:
            exact_norm += 1
        predictions.append(predicted)
        refs_multi.append(options)
        sources.append(spanish)
        examples.append(
            {
                "category": item.get("category", ""),
                "spanish": spanish,
                "expected": gold,
                "predicted": predicted,
                "exact_strict": exact_ok,
                "exact_normalized": norm_ok,
            }
        )

    total = len(test_items)
    bleu = bleu_metric.compute(predictions=predictions, references=refs_multi)

    rouge_scores = [
        _max_score(rouge_metric, pred, refs, "rougeL")
        for pred, refs in zip(predictions, refs_multi)
    ]
    meteor_scores = [
        _max_score(meteor_metric, pred, refs, "meteor")
        for pred, refs in zip(predictions, refs_multi)
    ]
    neural = score_neural_metrics(predictions, refs_multi, sources)
    for row, bert_f1, comet_value, bleurt_value in zip(
        examples,
        neural["bertscore_f1"],
        neural["comet"],
        neural["bleurt"],
    ):
        row["bertscore_f1"] = bert_f1
        row["comet"] = comet_value
        row["bleurt"] = bleurt_value

    return {
        "model_name": model_name,
        "task": "spanish_to_lsa_glosses",
        "format": "GGUF_Q4_K_M",
        "total_test_samples": total,
        "accuracy_strict_percent": round((exact_strict / total) * 100, 2) if total else 0.0,
        "accuracy_normalized_percent": round((exact_norm / total) * 100, 2) if total else 0.0,
        "bleu_score": round(bleu["bleu"] * 100, 2),
        "rouge_l_score": round((sum(rouge_scores) / total) * 100, 2) if total else 0.0,
        "meteor_score": round((sum(meteor_scores) / total) * 100, 2) if total else 0.0,
        "bertscore_f1": mean_percent(neural["bertscore_f1"]),
        "comet_score": mean_percent(neural["comet"]),
        "bleurt_score": mean_percent(neural["bleurt"]),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
        "evaluated_examples": examples,
    }


def copy_scores(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in SCORE_KEYS}
