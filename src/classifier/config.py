# ==========================================
# PATHS
# ==========================================
from pathlib import Path

# Absolutos: el proyecto debe poder ejecutarse desde cualquier directorio
_CLASSIFIER_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _CLASSIFIER_DIR / "weights"

WEIGHTS_PATH = str(_WEIGHTS_DIR / "tinyskeleton_best.pth")
CLASSES_PATH = str(_WEIGHTS_DIR / "mapeo_clases.json")
METRICS_PATH = str(_WEIGHTS_DIR / "metrics.json")

MAX_FRAMES = 32

SIGN_CLASSES = [
    "como",
    "cuando",
    "donde",
    "que",
    "quien",
    "si",
    "no",
    "cuantos",
    "bien",
    "mal",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "ñ",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "yo",
    "vos",
    "el_ella",
    "nosotros",
    "ellos",
    "hola",
    "chau",
    #"departamento",
    "lugar",
    "nombre",
    "apellido",
    "documento",
    "dia",
    "hora",
    "familia",
    "mama",
    "papa",
    "hermano_a",
    "tener",
    #"arma",
    "cuchillo",
    "brazo",
    "cara",
    "hijo_a",
    "numero",
    "años",
    "ojo",
    "esposo a",
    "casa",
    "calle",
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
    "plaza",
    "ahora_hoy",
    "ayer",
    #"golpear",
    "poder",
    #"sacar",
    "robar",
    #"pasar",
    "llevar",
    "tuyo",
    #"lastimar",
    "ver",
    "llamar",
    "repetir",
    "vivir",
    "vivir_en",
]

POSE_DIM = 33 * 3
HANDS_DIM = (21 * 3) * 2
FRAME_FEATURES_DIM = POSE_DIM + HANDS_DIM

# ==========================================
# VOICE
# ==========================================
VOICE = True

# ==========================================
# TINY SKELETON CLASSIFIER
# ==========================================
HIDDEN_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT_RATE = 0.3028748566702939

# ==========================================
# DATA AUGMENTATION AND TRAINING
# ==========================================
USE_DATA_AUGMENTATION = True
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 15
VIRTUAL_MULTIPLIER = 10
AUG_NOISE_STD = 0.02992555713844365
LR = 3.772811699894694e-05
WEIGHT_DECAY = 0.0003168710901337327
LABEL_SMOOTHING = 0.0014563065114717305
AUG_SCALE_RANGE = (0.85, 1.15)

# ==========================================
# REAL-TIME INFERENCE
# ==========================================
CONFIDENCE_THRESHOLD = 0.75

INFERENCE_COOLDOWN_SEC = 1.0 # Wait time after a prediction before accepting a new one

# Dynamic gesture detection thresholds
MOTION_PIXEL_THRESHOLD = 500
LANDMARK_MOTION_THRESHOLD = 0.008

# Static sign detection thresholds
STATIC_HANDS_FRAMES_TO_START = 4
STATIC_GESTURE_MOTION_THRESHOLD = 0.012

# Sign end detection thresholds
STILL_FRAMES_LIMIT = 10
CAPTURE_BUFFER_SIZE = 60
MISSING_HANDS_LIMIT = 12
MIN_CAPTURE_FRAMES = 5

# Modes: "auto" (dynamic + static), "dynamic", "static"
CAPTURE_MODE = "auto"

# ==========================================
# BUFFER (glosses → LLM)
# ==========================================
# Long pause without new glosses → close list and send to the LLM
UTTERANCE_PAUSE_SEC = 4.0
# Dedup temporal para glosas "other" (no dígito/letra de un carácter)
GLOSS_DEDUP_SEC = 1.0
# Letras consecutivas iguales permitidas (la 3ª+ se descarta). Dígitos: sin límite.
LETTER_MAX_CONSECUTIVE = 2

# Normalization key mapping_clases.json → token for the LLM
GLOSS_NORMALIZER = {
    "el_ella": "EL/ELLA",
    "esposo a": "ESPOSO/A",
    "ahora_hoy": "HOY",
    "vivir_en": "VIVIR-EN",
    "hermano_a": "HERMANO/A",
    "hijo_a": "HIJO/A",
    "años": "AÑOS",
    "ñ": "Ñ",
}

STATIC_SIGN_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "8",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M",
    "N", "ñ", "O", "P", "Q", "S", "T", "U", "V", "W", "X", "Y", "Z",
]