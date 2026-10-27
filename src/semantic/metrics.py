"""Puntúa predicciones de glosas. No se llama evaluate: ese nombre lo usa Hugging Face.

Léxicas: accuracy, BLEU, ROUGE-L, METEOR.
Neuronales: BERTScore, COMET, BLEURT, en metrics_neural.py.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gguf import predecir_glosas
from gloss import normalizar_para_metricas
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


def evaluar_pares(
    llm: Any,
    system_prompt: str,
    test_items: Sequence[dict[str, str]],
    refs_por_espanol: dict[str, list[str]],
    model_name: str,
) -> dict[str, Any]:
    import evaluate as hf_evaluate

    bleu_metric = hf_evaluate.load("bleu")
    rouge_metric = hf_evaluate.load("rouge")
    meteor_metric = hf_evaluate.load("meteor")

    predicciones: list[str] = []
    refs_multi: list[list[str]] = []
    fuentes: list[str] = []
    latencies_ms: list[float] = []
    ejemplos: list[dict[str, Any]] = []
    exact_strict = 0
    exact_norm = 0

    for item in test_items:
        spanish = item["spanish"]
        gold = item["glosses"]
        opciones = refs_por_espanol.get(spanish, [gold])
        predicho, latency_ms = predecir_glosas(llm, system_prompt, spanish)
        latencies_ms.append(latency_ms)
        exact_ok = predicho in opciones
        norm_ok = any(
            normalizar_para_metricas(predicho) == normalizar_para_metricas(ref) for ref in opciones
        )
        if exact_ok:
            exact_strict += 1
        if norm_ok:
            exact_norm += 1
        predicciones.append(predicho)
        refs_multi.append(opciones)
        fuentes.append(spanish)
        ejemplos.append(
            {
                "category": item.get("category", ""),
                "espanol": spanish,
                "esperado": gold,
                "predicho": predicho,
                "exact_strict": exact_ok,
                "exact_normalized": norm_ok,
            }
        )

    total = len(test_items)
    bleu = bleu_metric.compute(predictions=predicciones, references=refs_multi)

    rouge_scores = [
        _max_score(rouge_metric, pred, refs, "rougeL")
        for pred, refs in zip(predicciones, refs_multi)
    ]
    meteor_scores = [
        _max_score(meteor_metric, pred, refs, "meteor")
        for pred, refs in zip(predicciones, refs_multi)
    ]
    neural = score_neural_metrics(predicciones, refs_multi, fuentes)
    for row, bert_f1, comet_value, bleurt_value in zip(
        ejemplos,
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
        "ejemplos_evaluados": ejemplos,
    }


def copiar_puntajes(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in SCORE_KEYS}
