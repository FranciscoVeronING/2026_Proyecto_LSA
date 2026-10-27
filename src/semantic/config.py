"""Paths, hyperparameters, and the model list."""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "outputs"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"
DEFAULT_DATASET = SCRIPT_DIR / "dataset_glosses.json"
DEFAULT_TRIAL_DATASET = SCRIPT_DIR / "dataset_trial.json"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompts" / "sys_prompt.txt"

MAX_SEQ_LENGTH = 2048
# The system prompt is about 1600 tokens. 512 truncates the gloss and drops the answer.
TRAIN_SEQ_LENGTH = 2048
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
        "label": "Qwen2.5 0.5B (ultra light)",
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
        "label": "Qwen2.5 3B (higher capacity)",
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


def model_folder_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def gguf_dir_for(output_base: Path, model_name: str) -> Path:
    return output_base / f"{model_folder_name(model_name)}_gguf"


def select_models(ids: list[str] | None) -> list[dict]:
    """Filter MODELS. With no ids, return every model in list order."""
    if not ids:
        return list(MODELS)
    by_id = {model["id"]: model for model in MODELS}
    missing = [model_id for model_id in ids if model_id not in by_id]
    if missing:
        known = ", ".join(model["id"] for model in MODELS)
        raise ValueError(f"Unknown ids: {', '.join(missing)}. Valid ids: {known}")
    return [by_id[model_id] for model_id in ids]


def is_large_model(model_name: str) -> bool:
    return "3b" in model_name.lower()
