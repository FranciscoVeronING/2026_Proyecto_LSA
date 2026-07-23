import json
import os
import torch
from datasets import Dataset
import evaluate
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ==========================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
# Podés probar cambiando el modelo base aquí:
# - "unsloth/Qwen2.5-0.5B-Instruct"  (Ultra liviano)
# - "unsloth/Qwen2.5-1.5B-Instruct"
# - "unsloth/Llama-3.2-1B-Instruct"   (Alternativa de Meta)
# - Phi-3-mini-4k
# - SmolLM2-1.7B
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"

# Cargar métricas de Hugging Face
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

folder_safe_name = MODEL_NAME.replace("/", "_")
OUTPUT_DIR = f"./outputs/{folder_safe_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CARGAR PROMPTS Y EJEMPLOS EXTERNOS
# ==========================================
def cargar_prompts_y_ejemplos():
    with open("./prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt_base = f.read().strip()

    try:
        with open("./prompts/few_shot_examples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            few_shots = data.get("examples", [])
    except FileNotFoundError:
        few_shots = []

    system_prompt_completo = system_prompt_base
    if few_shots:
        system_prompt_completo += "\n\nEjemplos de traducción:"
        for ex in few_shots:
            glosas_str = " ".join(ex["glosses"])
            system_prompt_completo += f"\nGlosas: {glosas_str} -> Español: {ex['spanish']}"

    return system_prompt_completo


SYSTEM_PROMPT = cargar_prompts_y_ejemplos()
print("--- SYSTEM PROMPT CONSTRUIDO ---")
print(SYSTEM_PROMPT)
print("--------------------------------\n")

MAX_SEQ_LENGTH = 512

# ==========================================
# 3. CARGAR MODELO Y TOKENIZADOR
# ==========================================
print(f"Loading model: {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ==========================================
# 4. CARGAR Y FORMATO DEL DATASET
# ==========================================
with open("./dataset_glosas.json", "r", encoding="utf-8") as f:
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Glosas: {glosas_str}"},
            {"role": "assistant", "content": espanol},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
    return {"text": formatted_texts}


train_dataset = train_raw.map(format_prompts, batched=True)

# ==========================================
# 5. CONFIGURAR ENTRENAMIENTO
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
    ),
)

print(f"Iniciando entrenamiento para {MODEL_NAME}...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ==========================================
# 6. EVALUACIÓN DE PRECISIÓN EN EL TEST SET
# ==========================================
print("\n--- INICIANDO EVALUACIÓN EN EL CONJUNTO DE TEST NO VISTO ---")
FastLanguageModel.for_inference(model)

predicciones = []
referencias = []
exact_matches = 0

for item in test_raw:
    glosas_str = " ".join(item["glosses"])
    referencia_real = item["spanish"].strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Glosas: {glosas_str}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=64,
        use_cache=True,
        temperature=0.1,
    )

    prediccion_modelo = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    predicciones.append(prediccion_modelo)
    referencias.append(referencia_real)

    if prediccion_modelo.lower() == referencia_real.lower():
        exact_matches += 1

total_test = len(test_raw)
accuracy_exact = round((exact_matches / total_test) * 100, 2)

bleu_results = bleu_metric.compute(predictions=predicciones, references=[[r] for r in referencias])
bleu_score = round(bleu_results["bleu"] * 100, 2)

rouge_results = rouge_metric.compute(predictions=predicciones, references=referencias)
rouge_l_score = round(rouge_results["rougeL"] * 100, 2)

# ==========================================
# 7. GUARDAR RESULTADOS EN METRICS.JSON
# ==========================================
metrics_accuracy = {
    "model_name": MODEL_NAME,
    "total_test_samples": total_test,
    "accuracy_exact_match_percent": accuracy_exact,
    "bleu_score": bleu_score,
    "rouge_l_score": rouge_l_score,
    "ejemplos_evaluados": [
        {"glosas": " ".join(item["glosses"]), "esperado": ref, "predicho": pred}
        for item, ref, pred in zip(test_raw, referencias, predicciones)
    ],
}

metrics_filepath = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_filepath, "w", encoding="utf-8") as f:
    json.dump(metrics_accuracy, f, indent=4, ensure_ascii=False)

print("\n==========================================")
print(f"RESULTADOS DE EVALUACIÓN: {MODEL_NAME}")
print(f"Accuracy Exacta (Frase idéntica): {accuracy_exact}%")
print(f"BLEU Score (Calidad de traducción): {bleu_score} / 100")
print(f"ROUGE-L Score (Fluidez): {rouge_l_score} / 100")
print(f"Métricas guardadas en: {metrics_filepath}")
print("==========================================")
