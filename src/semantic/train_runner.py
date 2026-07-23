"""Logica reutilizable para entrenar un modelo (usada por train_all_models.py)."""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "outputs"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def model_folder_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def load_system_prompt() -> str:
    with open(SCRIPT_DIR / "prompts" / "sys_prompt.txt", encoding="utf-8") as f:
        system_prompt_base = f.read().strip()

    few_shot_path = SCRIPT_DIR / "prompts" / "few_shot_examples.json"
    few_shots = []
    if few_shot_path.exists():
        with open(few_shot_path, encoding="utf-8") as f:
            few_shots = json.load(f).get("examples", [])

    system_prompt = system_prompt_base
    if few_shots:
        system_prompt += "\n\nEjemplos de traducción:"
        for ex in few_shots:
            glosas_str = " ".join(ex["glosses"])
            system_prompt += f"\nGlosas: {glosas_str} -> Español: {ex['spanish']}"
    return system_prompt


def build_lora_model(model):
    common_kwargs = {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    }
    try:
        return FastLanguageModel.get_peft_model(
            model,
            target_modules=TARGET_MODULES,
            **common_kwargs,
        )
    except ValueError as exc:
        print(f"target_modules estándar falló ({exc}). Reintentando con all-linear...")
        return FastLanguageModel.get_peft_model(
            model,
            target_modules="all-linear",
            **common_kwargs,
        )


def save_loss_plot(log_history: list[dict], output_dir: Path) -> Path | None:
    points = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "loss" in entry and "step" in entry
    ]
    if not points:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado; se omite loss_plot.png")
        return None

    steps, losses = zip(*points)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="o", linewidth=2)
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "loss_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def evaluate_model(model, tokenizer, system_prompt: str, test_raw) -> dict:
    import evaluate

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA no disponible; la evaluación será muy lenta.")

    predicciones: list[str] = []
    referencias: list[str] = []
    exact_matches = 0

    for item in test_raw:
        glosas_str = " ".join(item["glosses"])
        referencia_real = item["spanish"].strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Glosas: {glosas_str}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=64,
            use_cache=True,
            temperature=0.1,
        )
        prediccion = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True).strip()
        predicciones.append(prediccion)
        referencias.append(referencia_real)
        if prediccion.lower() == referencia_real.lower():
            exact_matches += 1

    total_test = len(test_raw)
    bleu_results = bleu_metric.compute(predictions=predicciones, references=[[r] for r in referencias])
    rouge_results = rouge_metric.compute(predictions=predicciones, references=referencias)

    return {
        "total_test_samples": total_test,
        "accuracy_exact_match_percent": round((exact_matches / total_test) * 100, 2),
        "bleu_score": round(bleu_results["bleu"] * 100, 2),
        "rouge_l_score": round(rouge_results["rougeL"] * 100, 2),
        "ejemplos_evaluados": [
            {"glosas": " ".join(item["glosses"]), "esperado": ref, "predicho": pred}
            for item, ref, pred in zip(test_raw, referencias, predicciones)
        ],
    }


def train_single_model(
    model_name: str,
    output_base: Path | None = None,
    dataset_path: Path | None = None,
    max_steps: int = 60,
    max_seq_length: int = 512,
    verbose_prompt: bool = False,
) -> dict:
    output_base = output_base or DEFAULT_OUTPUT_BASE
    dataset_path = dataset_path or (SCRIPT_DIR / "dataset_glosas.json")
    output_dir = output_base / model_folder_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    print(f"\n{'=' * 60}")
    print(f"Modelo: {model_name}")
    print(f"Salida: {output_dir}")
    print(f"{'=' * 60}\n")

    system_prompt = load_system_prompt()
    if verbose_prompt:
        print("--- SYSTEM PROMPT ---")
        print(system_prompt)
        print("---------------------\n")

    print(f"Cargando modelo {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = build_lora_model(model)

    with open(dataset_path, encoding="utf-8") as f:
        raw_json = json.load(f)
    raw_data = raw_json.get("dataset", raw_json.get("examples", []))

    full_dataset = Dataset.from_list(raw_data)
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=3407)
    train_raw = split_dataset["train"]
    test_raw = split_dataset["test"]

    def format_prompts(examples):
        formatted_texts = []
        for glosas_list, espanol in zip(examples["glosses"], examples["spanish"]):
            glosas_str = " ".join(glosas_list)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Glosas: {glosas_str}"},
                {"role": "assistant", "content": espanol},
            ]
            formatted_texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        return {"text": formatted_texts}

    train_dataset = train_raw.map(format_prompts, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(output_dir / "checkpoints"),
            report_to="none",
        ),
    )

    print("Iniciando entrenamiento...")
    train_result = trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = list(trainer.state.log_history)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    plot_path = save_loss_plot(log_history, output_dir)
    eval_metrics = evaluate_model(model, tokenizer, system_prompt, test_raw)

    metrics = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "max_steps": max_steps,
        "train_samples": len(train_raw),
        "test_samples": len(test_raw),
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 2),
        "train_loss_final": round(train_result.metrics.get("train_loss", 0), 4),
        "loss_plot": str(plot_path) if plot_path else None,
        **eval_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n--- RESULTADOS ---")
    print(f"Accuracy exacta: {metrics['accuracy_exact_match_percent']}%")
    print(f"BLEU: {metrics['bleu_score']}")
    print(f"ROUGE-L: {metrics['rouge_l_score']}")
    print(f"Métricas: {metrics_path}")
    if plot_path:
        print(f"Gráfico: {plot_path}")

    del trainer, model, tokenizer, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics
