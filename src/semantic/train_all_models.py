"""Entrena secuencialmente los LLM candidatos, exporta a GGUF (Q4_K_M) y evalúa métricas reales sobre el binario cuantizado."""
from __future__ import annotations

# IMPORTANTE: unsloth debe importarse ANTES de trl, transformers o peft
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import evaluate
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# ==========================================
# CONSTANTES Y CONFIGURACIÓN BASE
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "outputs"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"
DEFAULT_DATASET = SCRIPT_DIR / "dataset_glosas.json"
DEFAULT_PROMPTS = SCRIPT_DIR / "prompts" / "sys_prompt.txt"

MAX_SEQ_LENGTH = 2048

MODELS = [
    {
        "id": "qwen2.5-0.5b",
        "name": "unsloth/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B (ultra liviano)",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "export_gguf": True,
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "unsloth/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "export_gguf": True,
    },
    {
        "id": "qwen2.5-3b",
        "name": "unsloth/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B (mayor capacidad)",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "export_gguf": True,
    },
    {
        "id": "llama-3.2-1b",
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "label": "Llama 3.2 1B (Meta)",
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "export_gguf": True,
    },
    {
        "id": "smollm2-1.7b",
        "name": "unsloth/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B",
        "chat_template": "chatml",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "export_gguf": True,
    },
]


def model_folder_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def normalizar_texto(texto: str) -> str:
    """Limpia signos de puntuación y espacios para una comparación justa."""
    texto = texto.lower().strip()
    texto = re.sub(r"[¿?¡!.,;\"]", "", texto)
    return " ".join(texto.split())


def limpiar_salida_deepseek(texto: str) -> str:
    """Elimina las etiquetas <think>...</think> y su contenido."""
    texto_limpio = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_limpio.strip()


def unificar_glosas_dactilologicas(glosas: list[str] | str) -> list[str]:
    """Compacta letras sueltas consecutivas en palabras capitalizadas y dígitos en números continuos.

    Reglas:
    - Letras: Se permite un máximo de 2 caracteres iguales consecutivos; 3 o más se truncan a 2.
    - Dígitos: Se preservan repeticiones arbitrarias sin límite (ej. 30111222).
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
            # Limita repeticiones a un máximo de 2 letras iguales consecutivas (case-insensitive)
            palabra = re.sub(r"([a-zA-ZñÑ])\1{2,}", r"\1\1", palabra)

            # Si es una única letra de vocabulario de señas como 'X' o 'Ñ', la preserva tal cual
            if len(palabra) == 1 and palabra.upper() in {"X", "Ñ"}:
                resultado.append(palabra.upper())
            else:
                resultado.append(palabra.capitalize())
            buffer_letras = []

        if buffer_digitos:
            # Los números se concatenan sin truncar repeticiones
            resultado.append("".join(buffer_digitos))
            buffer_digitos = []

    for g in glosas:
        # Letras individuales (A-Z, Ñ)
        if len(g) == 1 and (g.isalpha() or g in {"ñ", "Ñ"}):
            if buffer_digitos:
                vaciar_buffers()
            buffer_letras.append(g)
        # Dígitos individuales (0-9)
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
        description="Entrena los LLMs en secuencia, exporta a GGUF y evalúa métricas reales sobre el binario.",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Carpeta base de salida (cada modelo tendrá su subcarpeta).",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Ruta al dataset JSON para entrenamiento y evaluación.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPTS),
        help="Ruta al archivo con el System Prompt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas de entrenamiento (default: 5).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Pasos máximos de entrenamiento (anula epochs si es > 0).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Carpeta donde se guardan los logs de la corrida.",
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
        help="IDs opcionales a entrenar (ej: qwen2.5-0.5b llama-3.2-1b). Default: todos.",
    )
    return parser.parse_args()


def log_line(log_fp, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def load_summary(summary_path: Path) -> dict:
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {"runs": []}
    return {"runs": []}


def save_summary(summary_path: Path, summary: dict) -> None:
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def train_and_eval_single_model(
    model_cfg: dict,
    output_dir: Path,
    dataset_path: Path,
    system_prompt: str,
    epochs: int = 5,
    max_steps: int = -1,
    log_fp=None,
) -> dict:
    model_name = model_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar métricas
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # 2. Cargar Modelo y Tokenizer
    log_line(log_fp, f"Cargando {model_name} en 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=model_cfg.get("chat_template", "chatml"),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=target_modules,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 3. Preparar Dataset con Dactilología Unificada
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
        raw_data = raw_json.get("dataset", raw_json.get("examples", []))

    # Preprocesar glosas del dataset base
    processed_data = []
    for item in raw_data:
        glosas_originales = item["glosses"]
        glosas_unificadas = unificar_glosas_dactilologicas(glosas_originales)
        processed_data.append({
            "glosses": glosas_unificadas,
            "spanish": item["spanish"]
        })

    full_dataset = Dataset.from_list(processed_data)
    split_dataset = full_dataset.train_test_split(test_size=0.15, seed=3407)
    train_raw = split_dataset["train"]
    test_raw = split_dataset["test"]

    def format_prompts(examples):
        formatted_texts = []
        for glosas_list, espanol in zip(examples["glosses"], examples["spanish"]):
            glosas_str = " ".join(glosas_list) if isinstance(glosas_list, list) else glosas_list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Glosas: {glosas_str}"},
                {"role": "assistant", "content": espanol},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return {"text": formatted_texts}

    train_dataset = train_raw.map(format_prompts, batched=True)

    # 4. Configurar Trainer
    is_large_model = "3b" in model_name.lower()
    batch_size = 2 if is_large_model else 4
    grad_accum = 8 if is_large_model else 4
    seq_length = 512

    sft_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "warmup_steps": 10,
        "learning_rate": 3e-4 if is_large_model else 5e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
        "dataset_text_field": "text",
        "max_length": seq_length,
        "packing": False,
        "report_to": "none",
    }

    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps
    else:
        sft_kwargs["num_train_epochs"] = epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(**sft_kwargs),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=model_cfg.get("instruction_part", "<|im_start|>user\n"),
        response_part=model_cfg.get("response_part", "<|im_start|>assistant\n"),
    )

    # 5. Entrenar y Guardar Adaptadores
    log_line(log_fp, f"Iniciando entrenamiento para {model_name}...")
    final_loss = None
    train_result = trainer.train()
    if hasattr(train_result, "training_loss"):
        final_loss = train_result.training_loss

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Exportación a GGUF
    temp_gguf_dir = Path("D:/temp_gguf")
    temp_unsloth_out = Path("D:/temp_gguf_gguf")

    for p in (temp_gguf_dir, temp_unsloth_out):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    temp_gguf_dir.mkdir(parents=True, exist_ok=True)

    log_line(log_fp, "Fusionando pesos y exportando a GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(str(temp_gguf_dir), tokenizer, quantization_method="q4_k_m")

    final_gguf_dir = output_dir.parent / f"{output_dir.name}_gguf"
    final_gguf_dir.mkdir(parents=True, exist_ok=True)

    found_files = list(temp_gguf_dir.glob("*.gguf")) + list(temp_unsloth_out.glob("*.gguf"))
    for item in found_files:
        dest = final_gguf_dir / item.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))

    for p in (temp_gguf_dir, temp_unsloth_out):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    del model, tokenizer, trainer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 6. Cargar motor GGUF para evaluación
    from llama_cpp import Llama

    candidates = list(final_gguf_dir.glob("*Q4_K_M.gguf")) or list(final_gguf_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(f"No se encontró binario GGUF en {final_gguf_dir}")
    gguf_file = candidates[0]

    log_line(log_fp, f"Cargando motor GGUF para evaluación directa: {gguf_file.name}")

    llm = Llama(
        model_path=str(gguf_file),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )

    # 7. Inferencia y Evaluación en Test Set
    log_line(log_fp, "Iniciando evaluación multi-referencia sobre el binario GGUF...")
    glosas_a_referencias = defaultdict(list)
    for item in processed_data:
        key = " ".join(item["glosses"]) if isinstance(item["glosses"], list) else item["glosses"].strip()
        texto_ref = item["spanish"].strip()
        if texto_ref not in glosas_a_referencias[key]:
            glosas_a_referencias[key].append(texto_ref)

    predicciones = []
    lista_referencias_multiples = []
    referencias_principales = []
    latencies_ms = []
    exact_matches_strict = 0
    exact_matches_normalized = 0

    for item in test_raw:
        glosas_str = " ".join(item["glosses"]) if isinstance(item["glosses"], list) else item["glosses"].strip()
        referencia_real = item["spanish"].strip()
        opciones_validas = glosas_a_referencias[glosas_str]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Glosas: {glosas_str}"},
        ]

        start_time = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=24,
            temperature=0.0,
            repeat_penalty=1.05,
            stop=["\n", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>"],
        )
        latencies_ms.append((time.time() - start_time) * 1000)

        prediccion_modelo = response["choices"][0]["message"]["content"].strip()
        prediccion_modelo = limpiar_salida_deepseek(prediccion_modelo)

        predicciones.append(prediccion_modelo)
        lista_referencias_multiples.append(opciones_validas)
        referencias_principales.append(referencia_real)

        if any(prediccion_modelo == ref for ref in opciones_validas):
            exact_matches_strict += 1

        if any(normalizar_texto(prediccion_modelo) == normalizar_texto(ref) for ref in opciones_validas):
            exact_matches_normalized += 1

    del llm
    gc.collect()

    # 8. Métricas
    total_test = len(test_raw)
    accuracy_strict = round((exact_matches_strict / total_test) * 100, 2)
    accuracy_normalized = round((exact_matches_normalized / total_test) * 100, 2)

    bleu_results = bleu_metric.compute(predictions=predicciones, references=lista_referencias_multiples)
    bleu_score = round(bleu_results["bleu"] * 100, 2)

    rouge_scores_sample = [
        max(rouge_metric.compute(predictions=[pred], references=[r])["rougeL"] for r in refs)
        for pred, refs in zip(predicciones, lista_referencias_multiples)
    ]
    meteor_scores_sample = [
        max(meteor_metric.compute(predictions=[pred], references=[r])["meteor"] for r in refs)
        for pred, refs in zip(predicciones, lista_referencias_multiples)
    ]

    rouge_l_score = round((sum(rouge_scores_sample) / total_test) * 100, 2)
    meteor_score = round((sum(meteor_scores_sample) / total_test) * 100, 2)
    avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2)

    metrics_accuracy = {
        "model_name": model_name,
        "format": "GGUF_Q4_K_M",
        "total_test_samples": total_test,
        "train_loss_final": final_loss,
        "accuracy_strict_percent": accuracy_strict,
        "accuracy_normalized_percent": accuracy_normalized,
        "bleu_score": bleu_score,
        "rouge_l_score": rouge_l_score,
        "meteor_score": meteor_score,
        "avg_latency_ms": avg_latency_ms,
        "ejemplos_evaluados": [
            {
                "glosas": " ".join(item["glosses"]) if isinstance(item["glosses"], list) else item["glosses"],
                "esperado": ref,
                "predicho": pred,
            }
            for item, ref, pred in zip(test_raw, referencias_principales, predicciones)
        ],
    }

    metrics_filepath = output_dir / "metrics.json"
    metrics_filepath.write_text(json.dumps(metrics_accuracy, indent=4, ensure_ascii=False), encoding="utf-8")

    return metrics_accuracy


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base)
    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    prompt_path = Path(args.prompt_file)

    output_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_path.exists():
        print(f"Error: No se encontró el system prompt en {prompt_path}", file=sys.stderr)
        return 1
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

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
        log_line(log_fp, f"Inicio batch de {len(selected)} modelos con evaluación sobre GGUF")
        log_line(log_fp, f"Salida base: {output_base}")

        for index, model_cfg in enumerate(selected, start=1):
            log_line(log_fp, f"\n[{index}/{len(selected)}] Entrenando {model_cfg['label']} ({model_cfg['name']})")
            started = datetime.now(timezone.utc).isoformat()
            model_dir = output_base / model_folder_name(model_cfg["name"])

            try:
                metrics = train_and_eval_single_model(
                    model_cfg=model_cfg,
                    output_dir=model_dir,
                    dataset_path=dataset_path,
                    system_prompt=system_prompt,
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    log_fp=log_fp,
                )
                exit_code = 0
            except Exception as exc:
                failures += 1
                exit_code = 1
                metrics = None
                log_line(log_fp, f"FAIL {model_cfg['id']}: {exc}")
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                if not args.continue_on_error:
                    log_line(log_fp, "Abortando batch por error.")
                    break
                continue

            finished = datetime.now(timezone.utc).isoformat()

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
                    "accuracy_strict_percent": metrics["accuracy_strict_percent"],
                    "accuracy_normalized_percent": metrics["accuracy_normalized_percent"],
                    "bleu_score": metrics["bleu_score"],
                    "rouge_l_score": metrics["rouge_l_score"],
                    "meteor_score": metrics["meteor_score"],
                    "avg_latency_ms": metrics["avg_latency_ms"],
                    "train_loss_final": metrics["train_loss_final"],
                },
            }

            batch["results"].append(result)
            summary["runs"].append(result)
            save_summary(summary_path, summary)

            log_line(
                log_fp,
                f"OK {model_cfg['id']} (GGUF) | acc_norm={result['metrics']['accuracy_normalized_percent']}% "
                f"bleu={result['metrics']['bleu_score']} rouge={result['metrics']['rouge_l_score']} "
                f"meteor={result['metrics']['meteor_score']} latencia_real={result['metrics']['avg_latency_ms']}ms",
            )

        batch["finished_at"] = datetime.now(timezone.utc).isoformat()
        batch["failures"] = failures
        batch_path = output_base / f"training_batch_{run_stamp}.json"
        batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        log_line(log_fp, f"\nResumen batch: {batch_path}")
        log_line(log_fp, f"Resumen acumulado: {summary_path}")
        log_line(log_fp, f"Fin. Fallos: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())