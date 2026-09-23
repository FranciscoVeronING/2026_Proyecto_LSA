import os
from pathlib import Path

# Absolutos: el proyecto debe poder ejecutarse desde cualquier directorio
_SEMANTIC_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = _SEMANTIC_DIR / "outputs"
MODELS_DIR = _SEMANTIC_DIR / "models"
SORDO_GGUF_DIR = MODELS_DIR / "sordo"
OYENTE_GGUF_DIR = MODELS_DIR / "oyente"

# Único GGUF del servidor semántico (Llama 3.2 1B).
DEFAULT_MODEL_ID = "llama-3.2-1b"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

ADAPTER_PATH = str(OUTPUTS_DIR / "unsloth_Llama-3.2-1B-Instruct_gguf")
SYSTEM_PROMPT_PATH = str(_SEMANTIC_DIR / "prompts" / "sys_prompt.txt")
SYSTEM_PROMPT_OYENTE_PATH = str(_SEMANTIC_DIR / "prompts" / "sys_prompt_oyente.txt")
FEW_SHOTS_PATH = str(_SEMANTIC_DIR / "prompts" / "few_shots_examples.json")

# ILSA (clasificador) llama a este FastAPI. Vacío = sin traducción remota.
SEMANTIC_URL = (os.getenv("LSA_SEMANTIC_URL") or "").strip().rstrip("/")
SEMANTIC_TOKEN = (os.getenv("LSA_SEMANTIC_TOKEN") or "").strip()
# Release con un txt de una línea: la URL de ngrok. ILSA la baja sola.
SEMANTIC_URL_REPO = os.getenv("LSA_GGUF_REPO", "FranciscoVeronING/2026_Proyecto_LSA")
SEMANTIC_URL_TAG = os.getenv("LSA_GGUF_TAG", "ilsa-llama-1b")
SEMANTIC_URL_ASSET = "semantic_url.txt"
# Segundo .gguf opcional para oyente; si falta, se reusa el de sordo.
OYENTE_GGUF = (os.getenv("LSA_OYENTE_GGUF") or "").strip()



MAX_SEQ_LENGTH = 4096
N_CTX = 4096
# Siempre CPU. No hay soporte CUDA/Vulkan en el motor empaquetado.
N_GPU_LAYERS = 0

# Params de generación 
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
# 1.0 = sin penalización. Valores >1 desalientan repetir tokens del contexto,
# lo que perjudica copiar dígitos del enunciado (documentos, teléfonos).
REPETITION_PENALTY = 1.0
LOAD_IN_4BIT = True

# Últimos N turnos (signer + hearing) que la memoria conserva como contexto
CONVERSATION_HISTORY_SIZE = 10

# El adapter LoRA se entrenó con pares de un solo turno ({"glosses": [...],
# "spanish": "..."}), sin historial. Inyectar turnos previos lo aleja de esa
# distribución, así que el historial se puede apagar para comparar calidad
# (BLEU / ROUGE-L) contra el dataset de evaluación.
# Con la memoria vacía el prompt es idéntico al de entrenamiento; la diferencia
# aparece recién a partir del segundo enunciado.
USE_CONVERSATION_HISTORY = True