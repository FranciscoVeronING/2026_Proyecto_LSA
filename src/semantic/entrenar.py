"""Entrena los LLMs del catálogo, exporta GGUF Q4_K_M y evalúa el holdout 15%."""
from __future__ import annotations

# Unsloth tiene que importarse antes de trl, transformers o peft.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import argparse
import gc
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from glosas import coincide, texto_glosas, unificar_glosas_dactilologicas
from inferencia import cargar_gguf, traducir_gguf
from metricas import score_all
from modelos import (
    DATASET_GLOSAS,
    LOGS,
    MAX_SEQ_LENGTH,
    OUTPUTS,
    PROMPT,
    carpeta_modelo,
    elegir_modelos,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TEMP_GGUF = Path("D:/temp_gguf")
TEMP_GGUF_EXTRA = Path("D:/temp_gguf_gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el catálogo, exporta GGUF y evalúa el holdout.")
    parser.add_argument("--output-base", default=str(OUTPUTS))
    parser.add_argument("--dataset", default=str(DATASET_GLOSAS))
    parser.add_argument("--prompt-file", default=str(PROMPT))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Si es > 0, reemplaza a --epochs.")
    parser.add_argument("--log-dir", default=str(LOGS))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="IDs del catálogo. Default: todos.")
    return parser.parse_args()


def log_line(log_fp, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}


def preparar_ejemplos(dataset_path: Path) -> list[dict]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw.get("dataset", raw.get("examples", []))
    return [
        {
            "glosses": unificar_glosas_dactilologicas(item["glosses"]),
            "spanish": item["spanish"],
        }
        for item in items
    ]


def exportar_gguf(model, tokenizer, output_dir: Path, log_fp) -> Path:
    for carpeta in (TEMP_GGUF, TEMP_GGUF_EXTRA):
        if carpeta.exists():
            shutil.rmtree(carpeta, ignore_errors=True)
    TEMP_GGUF.mkdir(parents=True, exist_ok=True)

    log_line(log_fp, "Fusionando pesos y exportando a GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(str(TEMP_GGUF), tokenizer, quantization_method="q4_k_m")

    destino = output_dir.parent / f"{output_dir.name}_gguf"
    destino.mkdir(parents=True, exist_ok=True)
    for origen in list(TEMP_GGUF.glob("*.gguf")) + list(TEMP_GGUF_EXTRA.glob("*.gguf")):
        final = destino / origen.name
        if final.exists():
            final.unlink()
        shutil.move(str(origen), str(final))
    for carpeta in (TEMP_GGUF, TEMP_GGUF_EXTRA):
        if carpeta.exists():
            shutil.rmtree(carpeta, ignore_errors=True)
    return destino


def evaluar_holdout(gguf_dir: Path, test_raw, referencias: dict, system_prompt: str, log_fp) -> dict:
    candidatos = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
    if not candidatos:
        raise FileNotFoundError(f"No se encontró GGUF en {gguf_dir}")
    gguf = candidatos[0]
    log_line(log_fp, f"Cargando motor GGUF para evaluación directa: {gguf.name}")
    llm = cargar_gguf(gguf)

    predicciones = []
    listas_ref = []
    esperados = []
    fuentes = []
    latencias = []
    exactos = normalizados = 0
    try:
        for item in test_raw:
            glosas = texto_glosas(item["glosses"])
            esperado = item["spanish"].strip()
            opciones = referencias[glosas]
            prediccion, latencia = traducir_gguf(llm, system_prompt, glosas)
            predicciones.append(prediccion)
            listas_ref.append(opciones)
            esperados.append(esperado)
            fuentes.append(glosas)
            latencias.append(latencia)
            estricto, normalizado = coincide(prediccion, opciones)
            exactos += int(estricto)
            normalizados += int(normalizado)
    finally:
        del llm
        gc.collect()

    total = len(test_raw)
    puntajes = score_all(predicciones, listas_ref, fuentes)
    return {
        "total_test_samples": total,
        "accuracy_strict_percent": round(100.0 * exactos / total, 2),
        "accuracy_normalized_percent": round(100.0 * normalizados / total, 2),
        "bleu_score": puntajes["bleu_score"],
        "rouge_l_score": puntajes["rouge_l_score"],
        "meteor_score": puntajes["meteor_score"],
        "bertscore_f1": puntajes["bertscore_f1"],
        "comet_score": puntajes["comet_score"],
        "bleurt_score": puntajes["bleurt_score"],
        "avg_latency_ms": round(sum(latencias) / len(latencias), 2),
        "ejemplos_evaluados": [
            {"glosas": fuente, "esperado": esperado, "predicho": prediccion}
            for fuente, esperado, prediccion in zip(fuentes, esperados, predicciones)
        ],
    }


def entrenar_un_modelo(model_cfg: dict, output_dir: Path, dataset_path: Path, system_prompt: str, epochs: int, max_steps: int, log_fp) -> dict:
    nombre = model_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_line(log_fp, f"Cargando {nombre} en 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=nombre,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg.get("chat_template", "chatml"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    ejemplos = preparar_ejemplos(dataset_path)
    dataset = Dataset.from_list(ejemplos).train_test_split(test_size=0.15, seed=3407)
    train_raw, test_raw = dataset["train"], dataset["test"]

    def format_prompts(batch):
        textos = []
        for glosas, espanol in zip(batch["glosses"], batch["spanish"]):
            mensajes = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Glosas: {texto_glosas(glosas)}"},
                {"role": "assistant", "content": espanol},
            ]
            textos.append(tokenizer.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=False))
        return {"text": textos}

    train_dataset = train_raw.map(format_prompts, batched=True)
    grande = "3b" in nombre.lower()
    args = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2 if grande else 4,
        "gradient_accumulation_steps": 8 if grande else 4,
        "warmup_steps": 10,
        "learning_rate": 3e-4 if grande else 5e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
        "dataset_text_field": "text",
        "max_length": 512,
        "packing": False,
        "report_to": "none",
    }
    if max_steps > 0:
        args["max_steps"] = max_steps
    else:
        args["num_train_epochs"] = epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**args),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=model_cfg.get("instruction_part", "<|im_start|>user\n"),
        response_part=model_cfg.get("response_part", "<|im_start|>assistant\n"),
    )

    log_line(log_fp, f"Iniciando entrenamiento para {nombre}...")
    resultado = trainer.train()
    perdida = resultado.training_loss if hasattr(resultado, "training_loss") else None
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    gguf_dir = exportar_gguf(model, tokenizer, output_dir, log_fp)

    del model, tokenizer, trainer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    referencias = defaultdict(list)
    for item in ejemplos:
        clave = texto_glosas(item["glosses"])
        texto = item["spanish"].strip()
        if texto not in referencias[clave]:
            referencias[clave].append(texto)

    log_line(log_fp, "Iniciando evaluación multi-referencia sobre el binario GGUF...")
    metricas = evaluar_holdout(gguf_dir, test_raw, referencias, system_prompt, log_fp)
    metricas.update({"model_name": nombre, "format": "GGUF_Q4_K_M", "train_loss_final": perdida})
    (output_dir / "metrics.json").write_text(json.dumps(metricas, indent=4, ensure_ascii=False), encoding="utf-8")
    return metricas


def resumen_metricas(metricas: dict) -> dict:
    claves = (
        "accuracy_strict_percent",
        "accuracy_normalized_percent",
        "bleu_score",
        "rouge_l_score",
        "meteor_score",
        "bertscore_f1",
        "comet_score",
        "bleurt_score",
        "avg_latency_ms",
        "train_loss_final",
    )
    return {clave: metricas[clave] for clave in claves}


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    prompt_path = Path(args.prompt_file)
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        print(f"Error: no se encontró el system prompt en {prompt_path}", file=sys.stderr)
        return 1

    elegidos, faltantes = elegir_modelos(args.models)
    if faltantes:
        print(f"IDs desconocidos: {', '.join(faltantes)}", file=sys.stderr)
        return 1

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    sello = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"entrenar_{sello}.log"
    summary_path = output_base / "training_summary.json"
    summary = load_summary(summary_path)
    batch = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(log_path),
        "models_requested": [modelo["id"] for modelo in elegidos],
        "results": [],
    }
    fallos = 0

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Inicio batch de {len(elegidos)} modelos con evaluación sobre GGUF")
        log_line(log_fp, f"Salida base: {output_base}")
        for indice, modelo in enumerate(elegidos, start=1):
            log_line(log_fp, f"\n[{indice}/{len(elegidos)}] Entrenando {modelo['label']} ({modelo['name']})")
            iniciado = datetime.now(timezone.utc).isoformat()
            salida = output_base / carpeta_modelo(modelo["name"])
            try:
                metricas = entrenar_un_modelo(
                    modelo, salida, dataset_path, system_prompt, args.epochs, args.max_steps, log_fp
                )
            except Exception as exc:
                fallos += 1
                log_line(log_fp, f"FAIL {modelo['id']}: {exc}")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if not args.continue_on_error:
                    log_line(log_fp, "Abortando batch por error.")
                    break
                continue

            resultado = {
                "id": modelo["id"],
                "model_name": modelo["name"],
                "label": modelo["label"],
                "started_at": iniciado,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0,
                "output_dir": str(salida),
                "metrics_file": str(salida / "metrics.json"),
                "metrics": resumen_metricas(metricas),
            }
            batch["results"].append(resultado)
            summary["runs"].append(resultado)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            m = resultado["metrics"]
            log_line(
                log_fp,
                f"OK {modelo['id']} (GGUF) | acc_norm={m['accuracy_normalized_percent']}% "
                f"bleu={m['bleu_score']} rouge={m['rouge_l_score']} meteor={m['meteor_score']} "
                f"bert={m['bertscore_f1']} comet={m['comet_score']} bleurt={m['bleurt_score']} "
                f"latencia_real={m['avg_latency_ms']}ms",
            )

        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        batch["failures"] = fallos
        batch_path = output_base / f"entrenamiento_{sello}.json"
        batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        log_line(log_fp, f"\nResumen batch: {batch_path}")
        log_line(log_fp, f"Resumen acumulado: {summary_path}")
        log_line(log_fp, f"Fin. Fallos: {fallos}")

    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
