"""Catálogo de LLMs y rutas del módulo semántico."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
DATASET_GLOSAS = ROOT / "dataset_glosas.json"
DATASET_TRIAL = ROOT / "dataset_trial.json"
PROMPT = ROOT / "prompts" / "sys_prompt.txt"
EVAL_TRIAL = OUTPUTS / "eval_trial"

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 24
GGUF_STOP = ["\n", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>"]

# chat_template e instruction/response los usa solo el entrenamiento (Unsloth).
MODELOS = [
    {
        "id": "qwen2.5-0.5b",
        "name": "unsloth/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B (ultra liviano)",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "unsloth/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    {
        "id": "qwen2.5-3b",
        "name": "unsloth/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B (mayor capacidad)",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    {
        "id": "llama-3.2-1b",
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "label": "Llama 3.2 1B (Meta)",
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    {
        "id": "smollm2-1.7b",
        "name": "unsloth/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B",
        "chat_template": "chatml",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
]


def carpeta_modelo(model_name: str) -> str:
    return model_name.replace("/", "_")


def elegir_modelos(ids: list[str] | None) -> tuple[list[dict], list[str]]:
    """Devuelve los modelos pedidos y los IDs que no existen en el catálogo."""
    if not ids:
        return list(MODELOS), []
    pedidos = set(ids)
    elegidos = [modelo for modelo in MODELOS if modelo["id"] in pedidos]
    faltantes = sorted(pedidos - {modelo["id"] for modelo in elegidos})
    return elegidos, faltantes
