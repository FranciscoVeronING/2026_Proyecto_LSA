from pathlib import Path

# Absolutos: el proyecto debe poder ejecutarse desde cualquier directorio
_SEMANTIC_DIR = Path(__file__).resolve().parent

ADAPTER_PATH = str(_SEMANTIC_DIR / "adapter")
BASE_MODEL_ID = "unsloth/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT_PATH = str(_SEMANTIC_DIR / "prompts" / "sys_prompt.txt")
FEW_SHOTS_PATH = str(_SEMANTIC_DIR / "prompts" / "few_shots_examples.json")



MAX_SEQ_LENGTH = 2048

# Params de generación 
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
# 1.0 = sin penalización. Valores >1 desalientan repetir tokens del contexto,
# lo que perjudica copiar dígitos del enunciado (documentos, teléfonos).
REPETITION_PENALTY = 1.0
LOAD_IN_4BIT = True

USE_UNSLOTH = False  # True: usar modelo UnsloTH, False: usar modelo Qwen2.5-3B-Instruct

# Últimos N turnos (signer + hearing) que la memoria conserva como contexto
CONVERSATION_HISTORY_SIZE = 10

# El adapter LoRA se entrenó con pares de un solo turno ({"glosses": [...],
# "spanish": "..."}), sin historial. Inyectar turnos previos lo aleja de esa
# distribución, así que el historial se puede apagar para comparar calidad
# (BLEU / ROUGE-L) contra el dataset de evaluación.
# Con la memoria vacía el prompt es idéntico al de entrenamiento; la diferencia
# aparece recién a partir del segundo enunciado.
USE_CONVERSATION_HISTORY = True