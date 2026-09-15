"""Rutas, hiperparámetros y lista de modelos."""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "outputs"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"
DEFAULT_DATASET = SCRIPT_DIR / "dataset_glosas.json"
DEFAULT_TRIAL_DATASET = SCRIPT_DIR / "dataset_trial.json"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompts" / "sys_prompt.txt"
DEFAULT_FEW_SHOT_FILE = SCRIPT_DIR / "prompts" / "few_shot_examples.json"

MAX_SEQ_LENGTH = 2048
TRAIN_SEQ_LENGTH = 512
SPLIT_SEED = 3407
TEST_SIZE = 0.15
MAX_NEW_TOKENS = 48
GGUF_STOP = ["\n", "<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>"]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

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
    return model_name.replace("/", "_").replace(":", "_")


def is_large_model(model_name: str) -> bool:
    return "3b" in model_name.lower()
