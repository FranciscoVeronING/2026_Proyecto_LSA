"""Rutas y parámetros de inferencia. Sin descargas en runtime."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR / "models"
SIGN_DIR = MODELS_DIR / "sign"
HEARING_DIR = MODELS_DIR / "hearing"

PRIMARY_GGUF = "qwen2.5-1.5b.gguf"
BACKUP_GGUF = "llama-3.2-1b.gguf"

AUTH_TOKEN = (os.getenv("LSA_SEMANTIC_TOKEN") or "").strip()

_default_n_ctx = 2048 if os.getenv("PORT") else 4096
N_CTX = int(os.getenv("LSA_N_CTX") or _default_n_ctx)
N_BATCH = 512
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
REPETITION_PENALTY = 1.0
LOCAL_PORT = 8787
