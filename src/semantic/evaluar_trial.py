"""Compara los modelos ya entrenados sobre dataset_trial y guarda la comparativa."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from glosas import coincide, limpiar_salida, texto_glosas, unificar_glosas_dactilologicas
from inferencia import cargar_gguf, mensajes_glosas, traducir_gguf
from metricas import score_all
from modelos import (
    DATASET_TRIAL,
    EVAL_TRIAL,
    MAX_NEW_TOKENS,
    OUTPUTS,
    PROMPT,
    carpeta_modelo,
    elegir_modelos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa el catálogo sobre dataset_trial.")
    parser.add_argument("--output-base", default=str(OUTPUTS))
    parser.add_argument("--dataset", default=str(DATASET_TRIAL))
    parser.add_argument("--prompt-file", default=str(PROMPT))
    parser.add_argument("--report-dir", default=None, help="Default: <output-base>/eval_trial.")
    parser.add_argument("--models", nargs="*", default=None, help="IDs del catálogo. Default: todos.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def cargar_trial(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw.get("test_dataset", raw.get("dataset", raw.get("examples", [])))
    if not isinstance(items, list) or not items:
        raise ValueError(f"No se encontraron ejemplos en {dataset_path}")

    muestras = []
    for item in items:
        glosas = unificar_glosas_dactilologicas(item["glosses"])
        espanol = item["spanish"]
        if isinstance(espanol, str):
            referencias = [espanol.strip()]
        else:
            referencias = [texto.strip() for texto in espanol if str(texto).strip()]
        if not referencias:
            continue
        muestras.append(
            {
                "category": item.get("category", "sin_categoria"),
                "glosses": glosas,
                "glosas_str": texto_glosas(glosas),
                "references": referencias,
            }
        )
    return muestras


def resolver_checkpoint(output_base: Path, modelo: dict) -> dict:
    carpeta = carpeta_modelo(modelo["name"])
    gguf_dir = output_base / f"{carpeta}_gguf"
    if gguf_dir.is_dir():
        candidatos = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
        if candidatos:
            return {"backend": "gguf", "path": candidatos[0]}
    hf_dir = output_base / carpeta
    if (hf_dir / "config.json").exists():
        return {"backend": "causal_hf", "path": hf_dir}
    raise FileNotFoundError(f"No hay GGUF ni checkpoint HF para {modelo['id']} ({carpeta})")


def predecir_gguf(ruta: Path, system_prompt: str, muestras: list[dict]) -> tuple[list[str], list[float]]:
    llm = cargar_gguf(ruta)
    predicciones, latencias = [], []
    try:
        for muestra in muestras:
            texto, latencia = traducir_gguf(llm, system_prompt, muestra["glosas_str"])
            predicciones.append(texto)
            latencias.append(latencia)
    finally:
        del llm
        gc.collect()
    return predicciones, latencias


def predecir_hf(ruta: Path, system_prompt: str, muestras: list[dict]) -> tuple[list[str], list[float]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ruta), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ruta),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predicciones, latencias = [], []
    try:
        for muestra in muestras:
            encoded = tokenizer.apply_chat_template(
                mensajes_glosas(system_prompt, muestra["glosas_str"]),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {clave: valor.to(model.device) for clave, valor in encoded.items()}
            inicio = time.time()
            with torch.no_grad():
                salida = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latencias.append((time.time() - inicio) * 1000)
            largo = encoded["input_ids"].shape[1]
            texto = tokenizer.decode(salida[0][largo:], skip_special_tokens=True).strip()
            predicciones.append(limpiar_salida(texto.split("\n")[0]))
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predicciones, latencias


def medir(predicciones: list[str], muestras: list[dict], latencias: list[float]) -> dict:
    referencias = [muestra["references"] for muestra in muestras]
    puntajes = score_all(predicciones, referencias, [muestra["glosas_str"] for muestra in muestras])
    ejemplos = []
    exactos = normalizados = 0
    for pred, muestra, rouge, meteor, bert, comet, bleurt in zip(
        predicciones,
        muestras,
        puntajes["rouge_l"],
        puntajes["meteor"],
        puntajes["bertscore_raw"],
        puntajes["comet_raw"],
        puntajes["bleurt_raw"],
    ):
        estricto, normalizado = coincide(pred, muestra["references"])
        exactos += int(estricto)
        normalizados += int(normalizado)
        ejemplos.append(
            {
                "category": muestra["category"],
                "glosas": muestra["glosas_str"],
                "referencias": muestra["references"],
                "predicho": pred,
                "exact_strict": estricto,
                "exact_normalized": normalizado,
                "rouge_l": rouge,
                "meteor": meteor,
                "bertscore_f1": bert,
                "comet": comet,
                "bleurt": bleurt,
            }
        )

    por_categoria: dict[str, dict] = {}
    for fila in ejemplos:
        bolsa = por_categoria.setdefault(fila["category"], {"n": 0, "exact_normalized": 0, "rouge_l": 0.0})
        bolsa["n"] += 1
        bolsa["exact_normalized"] += int(fila["exact_normalized"])
        bolsa["rouge_l"] += fila["rouge_l"]

    total = len(muestras)
    return {
        "total_samples": total,
        "accuracy_strict_percent": round(100.0 * exactos / total, 2),
        "accuracy_normalized_percent": round(100.0 * normalizados / total, 2),
        "bleu_score": puntajes["bleu_score"],
        "rouge_l_score": puntajes["rouge_l_score"],
        "meteor_score": puntajes["meteor_score"],
        "bertscore_f1": puntajes["bertscore_f1"],
        "comet_score": puntajes["comet_score"],
        "bleurt_score": puntajes["bleurt_score"],
        "avg_latency_ms": round(sum(latencias) / len(latencias), 2) if latencias else None,
        "by_category": {
            categoria: {
                "n": datos["n"],
                "accuracy_normalized_percent": round(100.0 * datos["exact_normalized"] / datos["n"], 2),
                "rouge_l_score": round(100.0 * datos["rouge_l"] / datos["n"], 2),
            }
            for categoria, datos in por_categoria.items()
        },
        "ejemplos": ejemplos,
    }


def evaluar_modelo(modelo: dict, backend: dict, system_prompt: str, muestras: list[dict]) -> dict:
    if backend["backend"] == "gguf":
        predicciones, latencias = predecir_gguf(backend["path"], system_prompt, muestras)
    elif backend["backend"] == "causal_hf":
        predicciones, latencias = predecir_hf(backend["path"], system_prompt, muestras)
    else:
        raise ValueError(f"Backend desconocido: {backend['backend']}")
    metricas = medir(predicciones, muestras, latencias)
    metricas.update(
        {
            "id": modelo["id"],
            "model_name": modelo["name"],
            "label": modelo["label"],
            "backend": backend["backend"],
            "checkpoint": str(backend["path"]),
        }
    )
    return metricas


def clave_ranking(metricas: dict) -> tuple:
    return (
        metricas["accuracy_normalized_percent"],
        metricas["bleu_score"],
        metricas["rouge_l_score"],
        metricas["meteor_score"],
        metricas["bertscore_f1"],
        metricas["comet_score"],
        metricas["bleurt_score"],
        -(metricas["avg_latency_ms"] or 10**9),
    )


def armar_ejemplos(muestras: list[dict], resultados: list[dict]) -> list[dict]:
    filas = []
    for indice, muestra in enumerate(muestras):
        predicciones = []
        for metricas in resultados:
            ejemplo = metricas["ejemplos"][indice]
            predicciones.append(
                {
                    "id": metricas["id"],
                    "label": metricas["label"],
                    "predicho": ejemplo["predicho"],
                    "exact_normalized": ejemplo["exact_normalized"],
                }
            )
        filas.append(
            {
                "category": muestra["category"],
                "glosas": muestra["glosas_str"],
                "referencias": muestra["references"],
                "predictions": predicciones,
            }
        )
    return filas


def markdown(reporte: dict) -> str:
    ranking = reporte["ranking"]
    lineas = [
        "# Comparativa de modelos — dataset_trial",
        "",
        f"- Dataset: `{reporte['dataset']}`",
        f"- Ejemplos: {reporte['n_samples']}",
        f"- Fecha: {reporte['evaluated_at']}",
        "",
        "## Ranking",
        "",
        "| # | Modelo | Backend | Acc. norm. % | Acc. estricta % | BLEU | ROUGE-L | METEOR | BERTScore | COMET | BLEURT | Latencia ms |",
        "|---|--------|---------|--------------|-----------------|------|---------|--------|-----------|-------|--------|-------------|",
    ]
    for indice, fila in enumerate(ranking, start=1):
        latencia = fila.get("avg_latency_ms")
        lineas.append(
            f"| {indice} | {fila['label']} | {fila['backend']} | {fila['accuracy_normalized_percent']:.2f} | "
            f"{fila['accuracy_strict_percent']:.2f} | {fila['bleu_score']:.2f} | {fila['rouge_l_score']:.2f} | "
            f"{fila['meteor_score']:.2f} | {fila['bertscore_f1']:.2f} | {fila['comet_score']:.2f} | "
            f"{fila['bleurt_score']:.2f} | {latencia if latencia is not None else '-'} |"
        )
    if reporte.get("failures"):
        lineas += ["", "## Fallos", ""]
        lineas += [f"- `{fallo['id']}`: {fallo['error']}" for fallo in reporte["failures"]]

    lineas += ["", "## Predicciones por ejemplo", ""]
    for item in reporte.get("per_example", []):
        lineas += [f"### {item['category']}", "", f"- Glosas: `{item['glosas']}`", "- Referencias:"]
        lineas += [f"  - {ref}" for ref in item["referencias"]]
        lineas += ["", "| Modelo | Predicción | Exacta (norm.) |", "|--------|------------|----------------|"]
        for pred in item["predictions"]:
            marca = "sí" if pred["exact_normalized"] else "no"
            texto = pred["predicho"].replace("|", "\\|")
            lineas.append(f"| {pred['label']} | {texto} | {marca} |")
        lineas.append("")

    categorias = sorted({cat for fila in ranking for cat in fila.get("by_category", {})})
    if categorias:
        lineas += ["", "## Métricas por categoría", ""]
        lineas.append("| Categoría | " + " | ".join(fila["id"] for fila in ranking) + " |")
        lineas.append("|-----------|" + "|".join(["--------"] * len(ranking)) + "|")
        for categoria in categorias:
            celdas = []
            for fila in ranking:
                datos = fila.get("by_category", {}).get(categoria)
                celdas.append(f"{datos['accuracy_normalized_percent']:.0f}%" if datos else "-")
            lineas.append(f"| `{categoria}` | " + " | ".join(celdas) + " |")
    return "\n".join(lineas) + "\n"


def imprimir_tabla(resultados: list[dict]) -> None:
    def celda(valor, ancho=7) -> str:
        if valor is None:
            return "-".rjust(ancho)
        if isinstance(valor, float):
            return f"{valor:.2f}".rjust(ancho)
        return str(valor).rjust(ancho)

    print("\n" + "=" * 132)
    print("COMPARATIVA dataset_trial")
    print("=" * 132)
    print(
        f"{'#':>2}  {'modelo':<28} {'back':<10} {'acc_n':>7} {'acc_s':>7} "
        f"{'bleu':>7} {'rouge':>7} {'meteor':>7} {'bert':>7} {'comet':>7} {'bleurt':>7} {'lat_ms':>8}"
    )
    print("-" * 132)
    for indice, fila in enumerate(resultados, start=1):
        print(
            f"{indice:>2}  {fila['id']:<28} {fila['backend']:<10} "
            f"{celda(fila['accuracy_normalized_percent'])} {celda(fila['accuracy_strict_percent'])} "
            f"{celda(fila['bleu_score'])} {celda(fila['rouge_l_score'])} {celda(fila['meteor_score'])} "
            f"{celda(fila['bertscore_f1'])} {celda(fila['comet_score'])} {celda(fila['bleurt_score'])} "
            f"{celda(fila.get('avg_latency_ms'), 8)}"
        )
    print("=" * 132)


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt_file)
    report_dir = Path(args.report_dir) if args.report_dir else output_base / EVAL_TRIAL.name
    if not dataset_path.exists():
        print(f"Error: no existe el dataset {dataset_path}", file=sys.stderr)
        return 1
    if not prompt_path.exists():
        print(f"Error: no existe el system prompt {prompt_path}", file=sys.stderr)
        return 1

    elegidos, faltantes = elegir_modelos(args.models)
    if faltantes:
        print(f"IDs desconocidos: {', '.join(faltantes)}", file=sys.stderr)
        return 1

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    muestras = cargar_trial(dataset_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    resultados, fallos = [], []
    print(f"Dataset: {dataset_path} ({len(muestras)} ejemplos)")
    print(f"Modelos candidatos: {', '.join(modelo['id'] for modelo in elegidos)}")

    for indice, modelo in enumerate(elegidos, start=1):
        print(f"\n[{indice}/{len(elegidos)}] Evaluando {modelo['label']} ({modelo['id']})")
        try:
            backend = resolver_checkpoint(output_base, modelo)
            print(f"  backend={backend['backend']}  path={backend['path']}")
            metricas = evaluar_modelo(modelo, backend, system_prompt, muestras)
            resultados.append(metricas)
            print(
                f"  acc_norm={metricas['accuracy_normalized_percent']}%  bleu={metricas['bleu_score']}  "
                f"rouge={metricas['rouge_l_score']}  meteor={metricas['meteor_score']}  "
                f"bert={metricas['bertscore_f1']}  comet={metricas['comet_score']}  "
                f"bleurt={metricas['bleurt_score']}  lat={metricas['avg_latency_ms']}ms"
            )
        except Exception as exc:
            fallos.append({"id": modelo["id"], "error": str(exc)})
            print(f"  FAIL {modelo['id']}: {exc}")
            gc.collect()
            if args.stop_on_error:
                break

    resultados.sort(key=clave_ranking, reverse=True)
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporte = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "n_samples": len(muestras),
        "ranking": [{clave: valor for clave, valor in fila.items() if clave != "ejemplos"} for fila in resultados],
        "per_example": armar_ejemplos(muestras, resultados) if resultados else [],
        "failures": fallos,
        "details": resultados,
    }
    json_text = json.dumps(reporte, indent=2, ensure_ascii=False)
    md_text = markdown(reporte)
    json_path = report_dir / f"comparativa_trial_{sello}.json"
    md_path = report_dir / f"comparativa_trial_{sello}.md"
    for path, text in (
        (json_path, json_text),
        (md_path, md_text),
        (report_dir / "comparativa_trial_latest.json", json_text),
        (report_dir / "comparativa_trial_latest.md", md_text),
    ):
        path.write_text(text, encoding="utf-8")

    if resultados:
        imprimir_tabla(resultados)
        print(f"Mejor modelo: {resultados[0]['label']} ({resultados[0]['id']})")
    print(f"\nReporte JSON: {json_path}")
    print(f"Reporte Markdown: {md_path}")
    if fallos:
        print(f"Fallos: {len(fallos)}")
        return 1
    return 0 if resultados else 1


if __name__ == "__main__":
    raise SystemExit(main())
