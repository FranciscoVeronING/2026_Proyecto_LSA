"""BLEU / ROUGE / METEOR sobre GGUF. El módulo se llama metrics para no tapar `import evaluate`."""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from config import GGUF_STOP, MAX_NEW_TOKENS, MAX_SEQ_LENGTH
from gloss_format import formato_canonico, limpiar_salida_modelo, normalizar_para_metricas
from prompts import build_messages


def load_gguf_engine(gguf_path: Any):
    from llama_cpp import Llama

    return Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )


def predecir_glosas(llm: Any, system_prompt: str, spanish: str) -> tuple[str, float]:
    messages = build_messages(system_prompt, spanish)
    started = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        repeat_penalty=1.05,
        stop=GGUF_STOP,
    )
    latency_ms = (time.time() - started) * 1000
    raw = response["choices"][0]["message"]["content"].strip()
    predicho = formato_canonico(limpiar_salida_modelo(raw))
    return predicho, latency_ms


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
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
        "ejemplos_evaluados": ejemplos,
    }
