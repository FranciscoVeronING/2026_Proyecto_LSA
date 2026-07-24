import os
import json
import torch
from transformers import logging
from unsloth import FastLanguageModel

# Ocultar advertencias secundarias de Transformers en consola
logging.set_verbosity_error()

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO A PROBAR
# ==========================================
MODEL_FOLDER = "unsloth_Qwen2.5-3B-Instruct"  # Ajustá al nombre de la carpeta
MODEL_PATH = f"./outputs/{MODEL_FOLDER}/lora_adapter"

MAX_SEQ_LENGTH = 2048  # Amplitud suficiente para evitar truncamiento del system prompt

# ==========================================
# 2. CARGAR SYSTEM PROMPT Y FEW-SHOTS
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

# ==========================================
# 3. CARGAR MODELO Y TOKENIZADOR
# ==========================================
print(f"Cargando modelo desde: {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Activar modo de inferencia nativo
FastLanguageModel.for_inference(model)

# Configurar token de relleno para evitar avisos de attention_mask
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.unk_token or "<|endoftext|>"

print("¡Modelo listo para traducir en tiempo real!\n")

# ==========================================
# 4. BUCLE DE INTERACCIÓN EN TIEMPO REAL
# ==========================================
print("=" * 60)
print("TRADUCTOR LSA -> ESPAÑOL (MÓDULO SEMÁNTICO)")
print("Escribí las glosas separadas por espacio (o 'salir' para terminar).")
print("Ejemplo: YO CASA IR MAÑANA")
print("=" * 60 + "\n")

while True:
    try:
        entrada_glosas = input("\n[Glosas] > ").strip()
        
        if entrada_glosas.lower() in ["salir", "exit", "q"]:
            print("\n¡Hasta luego!")
            break

        if not entrada_glosas:
            continue

        # Estructurar mensaje para ChatML (System + User actual)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Glosas: {entrada_glosas}"}
        ]

        # Aplicar plantilla obteniendo el diccionario de tensores
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        # Generar traducción
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=64,
            use_cache=True,
            temperature=0.1,         # Temperatura baja para evitar alucinaciones
            repetition_penalty=1.2,   # Evita repetición de palabras
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Calcular longitud del prompt enviado para recortar solo la respuesta nueva
        prompt_len = model_inputs["input_ids"].shape[1]
        traduccion = tokenizer.decode(
            outputs[0][prompt_len:], 
            skip_special_tokens=True
        ).strip()

        print(f"[Español] > {traduccion}")

    except KeyboardInterrupt:
        print("\n¡Sesión finalizada!")
        break
    except Exception as e:
        print(f"\nError durante la generación: {e}")