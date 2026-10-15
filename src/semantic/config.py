import os
from pathlib import Path

_SEMANTIC_DIR = Path(__file__).resolve().parent
MODELS_DIR = _SEMANTIC_DIR / "models"
SORDO_GGUF_DIR = MODELS_DIR / "sordo"
OYENTE_GGUF_DIR = MODELS_DIR / "oyente"

# Nombres fijos. No se busca en Ollama, Hugging Face ni AppData.
PRIMARY_GGUF = "qwen2.5-1.5b.gguf"
BACKUP_GGUF = "llama-3.2-1b.gguf"

SYSTEM_PROMPT_PATH = str(_SEMANTIC_DIR / "prompts" / "sys_prompt.txt")
SYSTEM_PROMPT_OYENTE_PATH = str(_SEMANTIC_DIR / "prompts" / "sys_prompt_oyente.txt")

SEMANTIC_TOKEN = (os.getenv("LSA_SEMANTIC_TOKEN") or "").strip()

# En Cloud Run (PORT) un n_ctx 4096 × 4 motores OOM/cuelga el mmap.
# Un GGUF Q4 1.5B + 2048 alcanza para las glosas.
_default_n_ctx = 2048 if os.getenv("PORT") else 4096
N_CTX = int(os.getenv("LSA_N_CTX") or _default_n_ctx)
N_GPU_LAYERS = 0
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.0

CONVERSATION_HISTORY_SIZE = 10
USE_CONVERSATION_HISTORY = True
