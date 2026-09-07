"""Evalúa todos los modelos entrenados sobre dataset_trial y genera una comparativa.

Usa los binarios GGUF (Q4_K_M) cuando existen, alineado con train_all_models.py.
Si no hay GGUF, intenta cargar el checkpoint Hugging Face de la carpeta del modelo.
Aplica unificación de dactilología y números previo a la inferencia.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import evaluate

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "outputs"
DEFAULT_DATASET = SCRIPT_DIR / "dataset_trial.json"
DEFAULT_PROMPTS = SCRIPT_DIR / "prompts" / "sys_prompt.txt"
MAX_SEQ_LENGTH = 2048
MAX_TOKENS = 24
GGUF_STOP = ["\n", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>"]

MODELS = [
    {
        "id": "qwen2.5-0.5b",
        "name": "unsloth/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B (ultra liviano)",
        "kind": "causal",
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "unsloth/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B",
        "kind": "causal",
    },
    {
        "id": "qwen2.5-3b",
        "name": "unsloth/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B (mayor capacidad)",
        "kind": "causal",
    },
    {
        "id": "llama-3.2-1b",
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "label": "Llama 3.2 1B (Meta)",
        "kind": "causal",
    },
    {
        "id": "smollm2-1.7b",
        "name": "unsloth/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B",
        "kind": "causal",
    },
]


def model_folder_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"[¿?¡!.,;\"]", "", texto)
    return " ".join(texto.split())


def limpiar_salida(texto: str) -> str:
    texto_limpio = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_limpio.strip()


def unificar_glosas_dactilologicas(glosas: list[str] | str) -> list[str]:
    """Compacta letras sueltas consecutivas en palabras capitalizadas y dígitos en números continuos.

    Reglas:
    - Letras: Máximo 2 caracteres iguales consecutivos; 3 o más colapsan a 2.
    - Dígitos: Se preservan repeticiones arbitrarias completas.
    """
    if isinstance(glosas, str):
        glosas = glosas.split()

    resultado = []
    buffer_letras = []
    buffer_digitos = []

    def vaciar_buffers():
        nonlocal buffer_letras, buffer_digitos
        if buffer_letras:
            palabra = "".join(buffer_letras)
            palabra = re.sub(r"([a-zA-ZñÑ])\1{2,}", r"\1\1", palabra)
            if len(palabra) == 1 and palabra.upper() in {"X", "Ñ"}:
                resultado.append(palabra.upper())
            else:
                resultado.append(palabra.capitalize())
            buffer_letras = []

        if buffer_digitos:
            resultado.append("".join(buffer_digitos))
            buffer_digitos = []

    for g in glosas:
        if len(g) == 1 and (g.isalpha() or g in {"ñ", "Ñ"}):
            if buffer_digitos:
                vaciar_buffers()
            buffer_letras.append(g)
        elif len(g) == 1 and g.isdigit():
            if buffer_letras:
                vaciar_buffers()
            buffer_digitos.append(g)
        else:
            vaciar_buffers()
            resultado.append(g)

    vaciar_buffers()
    return resultado


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa todos los modelos sobre dataset_trial y genera una comparativa.",
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPTS))
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Carpeta de reportes (default: <output-base>/eval_trial).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="IDs opcionales a evaluar. Default: todos los que tengan checkpoint.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Si un modelo falla, continuar con el siguiente (default: sí).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abortar el batch si un modelo falla.",
    )
    return parser.parse_args()


def load_trial_dataset(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw.get("test_dataset", raw.get("dataset", raw.get("examples", [])))
    if not isinstance(items, list) or not items:
        raise ValueError(f"No se encontraron ejemplos en {dataset_path}")

    normalized = []
    for item in items:
        glosses_raw = item["glosses"]
        # Compacta deletreo y números con las mismas reglas del entrenamiento
        glosses_unificadas = unificar_glosas_dactilologicas(glosses_raw)
        glosas_str = " ".join(glosses_unificadas)

        spanish = item["spanish"]
        if isinstance(spanish, str):
            refs = [spanish.strip()]
        else:
            refs = [s.strip() for s in spanish if str(s).strip()]
        if not refs:
            continue
        normalized.append(
            {
                "category": item.get("category", "sin_categoria"),
                "glosses": glosses_unificadas,
                "glosas_str": glosas_str,
                "references": refs,
            }
        )
    return normalized


def find_gguf(output_base: Path, folder: str) -> Path | None:
    gguf_dir = output_base / f"{folder}_gguf"
    if not gguf_dir.is_dir():
        return None
    candidates = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
    return candidates[0] if candidates else None


def find_hf_dir(output_base: Path, folder: str) -> Path | None:
    model_dir = output_base / folder
    if (model_dir / "config.json").exists():
        return model_dir
    return None


def resolve_backend(output_base: Path, model_cfg: dict) -> dict:
    folder = model_folder_name(model_cfg["name"])
    if model_cfg["kind"] == "seq2seq":
        hf_dir = find_hf_dir(output_base, folder)
        if hf_dir is None:
            raise FileNotFoundError(f"No hay checkpoint seq2seq en {output_base / folder}")
        return {"backend": "seq2seq", "path": hf_dir}

    gguf = find_gguf(output_base, folder)
    if gguf is not None:
        return {"backend": "gguf", "path": gguf}

    hf_dir = find_hf_dir(output_base, folder)
    if hf_dir is not None:
        return {"backend": "causal_hf", "path": hf_dir}

    raise FileNotFoundError(
        f"No se encontró GGUF ni checkpoint HF para {model_cfg['id']} "
        f"(buscado: {folder}_gguf / {folder})"
    )


def compute_metrics(predicciones: list[str], samples: list[dict], latencies_ms: list[float]) -> dict:
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    total = len(samples)
    lista_refs = [s["references"] for s in samples]
    exact_strict = 0
    exact_norm = 0
    per_sample = []

    for pred, sample in zip(predicciones, samples):
        refs = sample["references"]
        hit_strict = any(pred == ref for ref in refs)
        hit_norm = any(normalizar_texto(pred) == normalizar_texto(ref) for ref in refs)
        if hit_strict:
            exact_strict += 1
        if hit_norm:
            exact_norm += 1

        rouge_scores = [
            rouge_metric.compute(predictions=[pred], references=[r])["rougeL"] for r in refs
        ]
        meteor_scores = [
            meteor_metric.compute(predictions=[pred], references=[r])["meteor"] for r in refs
        ]
        per_sample.append(
            {
                "category": sample["category"],
                "glosas": sample["glosas_str"],
                "referencias": refs,
                "predicho": pred,
                "exact_strict": hit_strict,
                "exact_normalized": hit_norm,
                "rouge_l": max(rouge_scores) if rouge_scores else 0.0,
                "meteor": max(meteor_scores) if meteor_scores else 0.0,
            }
        )

    bleu_results = bleu_metric.compute(predictions=predicciones, references=lista_refs)
    rouge_l = sum(x["rouge_l"] for x in per_sample) / total
    meteor = sum(x["meteor"] for x in per_sample) / total

    by_category: dict[str, dict] = {}
    for row in per_sample:
        cat = row["category"]
        bucket = by_category.setdefault(cat, {"n": 0, "exact_normalized": 0, "rouge_l": 0.0})
        bucket["n"] += 1
        bucket["exact_normalized"] += int(row["exact_normalized"])
        bucket["rouge_l"] += row["rouge_l"]

    category_metrics = {
        cat: {
            "n": data["n"],
            "accuracy_normalized_percent": round(100.0 * data["exact_normalized"] / data["n"], 2),
            "rouge_l_score": round(100.0 * data["rouge_l"] / data["n"], 2),
        }
        for cat, data in by_category.items()
    }

    return {
        "total_samples": total,
        "accuracy_strict_percent": round(100.0 * exact_strict / total, 2),
        "accuracy_normalized_percent": round(100.0 * exact_norm / total, 2),
        "bleu_score": round(bleu_results["bleu"] * 100, 2),
        "rouge_l_score": round(rouge_l * 100, 2),
        "meteor_score": round(meteor * 100, 2),
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else None,
        "by_category": category_metrics,
        "ejemplos": per_sample,
    }


def eval_gguf(gguf_path: Path, system_prompt: str, samples: list[dict]) -> tuple[list[str], list[float]]:
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )
    predicciones = []
    latencies_ms = []
    try:
        for sample in samples:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Glosas: {sample['glosas_str']}"},
            ]
            start = time.time()
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                repeat_penalty=1.05,
                stop=GGUF_STOP,
            )
            latencies_ms.append((time.time() - start) * 1000)
            prediccion = response["choices"][0]["message"]["content"].strip()
            predicciones.append(limpiar_salida(prediccion))
    finally:
        del llm
        gc.collect()
    return predicciones, latencies_ms


def eval_causal_hf(model_dir: Path, system_prompt: str, samples: list[dict]) -> tuple[list[str], list[float]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predicciones = []
    latencies_ms = []
    try:
        for sample in samples:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Glosas: {sample['glosas_str']}"},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latencies_ms.append((time.time() - start) * 1000)
            prompt_len = encoded["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            predicciones.append(limpiar_salida(text.split("\n")[0]))
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predicciones, latencies_ms


def eval_seq2seq(model_dir: Path, samples: list[dict]) -> tuple[list[str], list[float]]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    predicciones = []
    latencies_ms = []
    try:
        for sample in samples:
            source = f"traducir glosas a español: {sample['glosas_str']}"
            encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(**encoded, max_new_tokens=MAX_TOKENS)
            latencies_ms.append((time.time() - start) * 1000)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            predicciones.append(limpiar_salida(text.split("\n")[0]))
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predicciones, latencies_ms


def eval_one_model(model_cfg: dict, backend: dict, system_prompt: str, samples: list[dict]) -> dict:
    if backend["backend"] == "gguf":
        preds, lats = eval_gguf(backend["path"], system_prompt, samples)
    elif backend["backend"] == "causal_hf":
        preds, lats = eval_causal_hf(backend["path"], system_prompt, samples)
    elif backend["backend"] == "seq2seq":
        preds, lats = eval_seq2seq(backend["path"], samples)
    else:
        raise ValueError(f"Backend desconocido: {backend['backend']}")

    metrics = compute_metrics(preds, samples, lats)
    metrics.update(
        {
            "id": model_cfg["id"],
            "model_name": model_cfg["name"],
            "label": model_cfg["label"],
            "backend": backend["backend"],
            "checkpoint": str(backend["path"]),
        }
    )
    return metrics


def fmt(value, width=8) -> str:
    if value is None:
        return str("-").rjust(width)
    if isinstance(value, float):
        return f"{value:.2f}".rjust(width)
    return str(value).rjust(width)


def build_markdown(report: dict) -> str:
    ranked = report["ranking"]
    lines = [
        "# Comparativa de modelos — dataset_trial",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Ejemplos: {report['n_samples']}",
        f"- Fecha: {report['evaluated_at']}",
        "",
        "## Ranking",
        "",
        "| # | Modelo | Backend | Acc. norm. % | Acc. estricta % | BLEU | ROUGE-L | METEOR | Latencia ms |",
        "|---|--------|---------|--------------|-----------------|------|---------|--------|-------------|",
    ]
    for i, row in enumerate(ranked, start=1):
        m = row
        lines.append(
            f"| {i} | {m['label']} | {m['backend']} | {m['accuracy_normalized_percent']:.2f} | "
            f"{m['accuracy_strict_percent']:.2f} | {m['bleu_score']:.2f} | {m['rouge_l_score']:.2f} | "
            f"{m['meteor_score']:.2f} | {m.get('avg_latency_ms') if m.get('avg_latency_ms') is not None else '-'} |"
        )

    if report.get("failures"):
        lines += ["", "## Fallos", ""]
        for fail in report["failures"]:
            lines.append(f"- `{fail['id']}`: {fail['error']}")

    lines += ["", "## Predicciones por ejemplo", ""]
    samples_index = report.get("per_example", [])
    for item in samples_index:
        lines.append(f"### {item['category']}")
        lines.append("")
        lines.append(f"- Glosas: `{item['glosas']}`")
        lines.append("- Referencias:")
        for ref in item["referencias"]:
            lines.append(f"  - {ref}")
        lines.append("")
        lines.append("| Modelo | Predicción | Exacta (norm.) |")
        lines.append("|--------|------------|----------------|")
        for pred in item["predictions"]:
            mark = "sí" if pred["exact_normalized"] else "no"
            safe = pred["predicho"].replace("|", "\\|")
            lines.append(f"| {pred['label']} | {safe} | {mark} |")
        lines.append("")

    lines += ["", "## Métricas por categoría", ""]
    categories = sorted({cat for m in ranked for cat in m.get("by_category", {})})
    if categories:
        header = "| Categoría | " + " | ".join(m["id"] for m in ranked) + " |"
        sep = "|-----------|" + "|".join(["--------"] * len(ranked)) + "|"
        lines += [header, sep]
        for cat in categories:
            cells = []
            for m in ranked:
                cat_m = m.get("by_category", {}).get(cat)
                cells.append(f"{cat_m['accuracy_normalized_percent']:.0f}%" if cat_m else "-")
            lines.append(f"| `{cat}` | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def print_table(results: list[dict]) -> None:
    print()
    print("=" * 108)
    print("COMPARATIVA dataset_trial")
    print("=" * 108)
    header = (
        f"{'#':>2}  {'modelo':<28} {'back':<10} {'acc_n':>7} {'acc_s':>7} "
        f"{'bleu':>7} {'rouge':>7} {'meteor':>7} {'lat_ms':>8}"
    )
    print(header)
    print("-" * 108)
    for i, m in enumerate(results, start=1):
        print(
            f"{i:>2}  {m['id']:<28} {m['backend']:<10} "
            f"{fmt(m['accuracy_normalized_percent'], 7)} {fmt(m['accuracy_strict_percent'], 7)} "
            f"{fmt(m['bleu_score'], 7)} {fmt(m['rouge_l_score'], 7)} {fmt(m['meteor_score'], 7)} "
            f"{fmt(m.get('avg_latency_ms'), 8)}"
        )
    print("=" * 108)


def rank_key(metrics: dict) -> tuple:
    return (
        metrics["accuracy_normalized_percent"],
        metrics["bleu_score"],
        metrics["rouge_l_score"],
        metrics["meteor_score"],
        -(metrics["avg_latency_ms"] or 10**9),
    )


def build_per_example(samples: list[dict], results: list[dict]) -> list[dict]:
    rows = []
    for idx, sample in enumerate(samples):
        preds = []
        for model_metrics in results:
            ejemplo = model_metrics["ejemplos"][idx]
            preds.append(
                {
                    "id": model_metrics["id"],
                    "label": model_metrics["label"],
                    "predicho": ejemplo["predicho"],
                    "exact_normalized": ejemplo["exact_normalized"],
                }
            )
        rows.append(
            {
                "category": sample["category"],
                "glosas": sample["glosas_str"],
                "referencias": sample["references"],
                "predictions": preds,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt_file)
    report_dir = Path(args.report_dir) if args.report_dir else output_base / "eval_trial"
    continue_on_error = not args.stop_on_error

    if not dataset_path.exists():
        print(f"Error: no existe el dataset {dataset_path}", file=sys.stderr)
        return 1
    if not prompt_path.exists():
        print(f"Error: no existe el system prompt {prompt_path}", file=sys.stderr)
        return 1

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    samples = load_trial_dataset(dataset_path)

    selected = MODELS
    if args.models:
        wanted = set(args.models)
        selected = [m for m in MODELS if m["id"] in wanted]
        missing = wanted - {m["id"] for m in selected}
        if missing:
            print(f"IDs desconocidos: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []

    print(f"Dataset: {dataset_path} ({len(samples)} ejemplos)")
    print(f"Modelos candidatos: {', '.join(m['id'] for m in selected)}")

    for index, model_cfg in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] Evaluando {model_cfg['label']} ({model_cfg['id']})")
        try:
            backend = resolve_backend(output_base, model_cfg)
            print(f"  backend={backend['backend']}  path={backend['path']}")
            metrics = eval_one_model(model_cfg, backend, system_prompt, samples)
            results.append(metrics)
            print(
                f"  acc_norm={metrics['accuracy_normalized_percent']}%  "
                f"bleu={metrics['bleu_score']}  rouge={metrics['rouge_l_score']}  "
                f"meteor={metrics['meteor_score']}  lat={metrics['avg_latency_ms']}ms"
            )
        except Exception as exc:
            failures.append({"id": model_cfg["id"], "error": str(exc)})
            print(f"  FAIL {model_cfg['id']}: {exc}")
            gc.collect()
            if not continue_on_error:
                break

    results.sort(key=rank_key, reverse=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "n_samples": len(samples),
        "ranking": [
            {k: v for k, v in m.items() if k != "ejemplos"}
            for m in results
        ],
        "per_example": build_per_example(samples, results) if results else [],
        "failures": failures,
        "details": results,
    }

    json_path = report_dir / f"comparativa_trial_{stamp}.json"
    md_path = report_dir / f"comparativa_trial_{stamp}.md"
    latest_json = report_dir / "comparativa_trial_latest.json"
    latest_md = report_dir / "comparativa_trial_latest.md"

    json_text = json.dumps(report, indent=2, ensure_ascii=False)
    md_text = build_markdown(report)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    if results:
        print_table(results)
        print(f"Mejor modelo: {results[0]['label']} ({results[0]['id']})")
    print(f"\nReporte JSON: {json_path}")
    print(f"Reporte Markdown: {md_path}")
    if failures:
        print(f"Fallos: {len(failures)}")
        return 1
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())