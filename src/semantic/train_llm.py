import re
import time
import json
import os

import evaluate
import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
# Podés probar cambiando el modelo base aquí:
# - "unsloth/Qwen2.5-0.5B-Instruct"  (Ultra liviano)
# - "unsloth/Qwen2.5-1.5B-Instruct"
# - "unsloth/Llama-3.2-1B-Instruct"   (Alternativa de Meta)
# - Phi-3-mini-4k
# - SmolLM2-1.7B
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

# Cargar métricas de Hugging Face
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

def normalizar_texto(texto):
    """Limpia signos de puntuación y espacios para una comparación justa."""
    texto = texto.lower().strip()
    texto = re.sub(r"[¿?¡!.,;\"]", "", texto)
    return " ".join(texto.split())

folder_safe_name = MODEL_NAME.replace("/", "_")
OUTPUT_DIR = f"src/semantic/outputs/{folder_safe_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CARGAR PROMPTS Y EJEMPLOS EXTERNOS
# ==========================================
def cargar_prompts():
    with open("src/semantic/prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt_base = f.read().strip()

    return system_prompt_base


SYSTEM_PROMPT = cargar_prompts()
print("--- SYSTEM PROMPT CONSTRUIDO ---")
print(SYSTEM_PROMPT)
print("--------------------------------\n")

MAX_SEQ_LENGTH = 2048

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
        "embed_tokens",
        "lm_head"
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ==========================================
# 4. CARGAR Y FORMATO DEL DATASET
# ==========================================
with open("src/semantic/dataset_glosas.json", "r", encoding="utf-8") as f:
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
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        num_train_epochs=10, # max_step = 60
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(f"Iniciando entrenamiento para {MODEL_NAME}...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ==========================================
# 6. EVALUACIÓN DE PRECISIÓN EN EL TEST SET
# ==========================================
from collections import defaultdict

print("\n--- INICIANDO EVALUACIÓN EN EL CONJUNTO DE TEST NO VISTO ---")
FastLanguageModel.for_inference(model)

# 1. Crear mapeo con todas las traducciones válidas por conjunto de glosas
glosas_a_referencias = defaultdict(list)
for item in raw_data:
    key = " ".join(item["glosses"])
    texto_ref = item["spanish"].strip()
    if texto_ref not in glosas_a_referencias[key]:
        glosas_a_referencias[key].append(texto_ref)

predicciones = []
lista_referencias_multiples = []
referencias_principales = []
exact_matches_strict = 0
exact_matches_normalized = 0
latencies_ms = []

def limpiar_salida_deepseek(texto: str) -> str:
    """Elimina las etiquetas <think>...</think> y su contenido, dejando solo la respuesta final."""
    texto_limpio = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_limpio.strip()

# 2. Loop de evaluación con inferencia
for item in test_raw:
    glosas_str = " ".join(item["glosses"])
    referencia_real = item["spanish"].strip()
    opciones_validas = glosas_a_referencias[glosas_str]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Glosas: {glosas_str}"},
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    # Medir latencia de inferencia
    start_time = time.time()
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=64,     
        do_sample=False,
        use_cache=True,
        temperature=None,
        top_p=None,
    )
    end_time = time.time()

    latencies_ms.append((end_time - start_time) * 1000)

    # Decodificar salida
    prediccion_raw = tokenizer.decode(
        outputs[0][model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    prediccion_modelo = limpiar_salida_deepseek(prediccion_raw)

    predicciones.append(prediccion_modelo)
    lista_referencias_multiples.append(opciones_validas)
    referencias_principales.append(referencia_real)

    # Coincidencia con CUALQUIERA de las variantes válidas
    if any(prediccion_modelo == ref for ref in opciones_validas):
        exact_matches_strict += 1

    if any(normalizar_texto(prediccion_modelo) == normalizar_texto(ref) for ref in opciones_validas):
        exact_matches_normalized += 1

# ==========================================
# CÁLCULO DE MÉTRICAS
# ==========================================
total_test = len(test_raw)

# Accuracies
accuracy_strict = round((exact_matches_strict / total_test) * 100, 2)
accuracy_normalized = round((exact_matches_normalized / total_test) * 100, 2)

# BLEU Score (soporta múltiples referencias por muestra)
bleu_results = bleu_metric.compute(
    predictions=predicciones, 
    references=lista_referencias_multiples
)
bleu_score = round(bleu_results["bleu"] * 100, 2)

# ROUGE-L Score
rouge_results = rouge_metric.compute(
    predictions=predicciones, 
    references=referencias_principales
)
rouge_l_score = round(rouge_results["rougeL"] * 100, 2)

# METEOR Score
meteor_results = meteor_metric.compute(
    predictions=predicciones, 
    references=referencias_principales
)
meteor_score = round(meteor_results["meteor"] * 100, 2)

# Latencia Promedio
avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2)

# ==========================================
# 7. GUARDAR RESULTADOS EN METRICS.JSON
# ==========================================
metrics_accuracy = {
    "model_name": MODEL_NAME,
    "total_test_samples": total_test,
    "accuracy_strict_percent": accuracy_strict,
    "accuracy_normalized_percent": accuracy_normalized,
    "bleu_score": bleu_score,
    "rouge_l_score": rouge_l_score,
    "meteor_score": meteor_score,
    "avg_latency_ms": avg_latency_ms,
    "ejemplos_evaluados": [
        {"glosas": " ".join(item["glosses"]), "esperado": ref, "predicho": pred}
        for item, ref, pred in zip(test_raw, referencias_principales, predicciones)
    ],
}

metrics_filepath = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_filepath, "w", encoding="utf-8") as f:
    json.dump(metrics_accuracy, f, indent=4, ensure_ascii=False)

print("\n==========================================")
print(f"RESULTADOS DE EVALUACIÓN: {MODEL_NAME}")
print(f"Accuracy Estricta: {accuracy_strict}%")
print(f"Accuracy Normalizada (sin puntuación): {accuracy_normalized}%")
print(f"BLEU Score (Precisión n-gramas): {bleu_score} / 100")
print(f"ROUGE-L Score (Fluidez estructural): {rouge_l_score} / 100")
print(f"METEOR Score (Manejo de sinónimos): {meteor_score} / 100")
print(f"Latencia promedio por oración: {avg_latency_ms} ms")
print(f"Métricas guardadas en: {metrics_filepath}")
print("==========================================")