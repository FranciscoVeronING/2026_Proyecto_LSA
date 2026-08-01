
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_VIDEOS_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_NPY_DIR = os.path.join(BASE_DIR, "dataset_hand_pose")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "src", "model")
WEIGHTS_PATH = os.path.join(BASE_DIR, "src", "model", "islr-fp16-192-8-seed_all42-foldall-last.h5")

NUM_CLASSES = 94
SAMPLES_PER_CLASS = 50

SIGN_CLASSES = [
    "como",
    "cuando",
    "donde",
    "que",
    "quienes",
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
    "esposo_a",
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
    "vivir_en"
]

POSE_DIM = 33 * 3
HANDS_DIM = (21 * 3) * 2
FRAME_FEATURES_DIM = POSE_DIM + HANDS_DIM

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FRAMES = 16

SIGN_TO_INDEX = {sign: idx for idx, sign in enumerate(SIGN_CLASSES)}
INDEX_TO_SIGN = {idx: sign for idx, sign in enumerate(SIGN_CLASSES)}

BATCH_SIZE = 16
EPOCHS_BASE = 15
EPOCHS = 999
LEARNING_RATE_TRANSFER = 0.0005639974395199206
LEARNING_RATE_FINE_TUNING = 1e-5
PATIENCE = 15
VAL_SIZE = 0.2



# ==========================================
# INFERENCIA EN TIEMPO REAL (WEBCAM)
# ==========================================
CONFIDENCE_THRESHOLD = 0.85

# --- Captura dinámica (señas con movimiento) ---
MOTION_PIXEL_THRESHOLD = 500
LANDMARK_MOTION_THRESHOLD = 0.008

# --- Captura estática (letras, números, poses cortas) ---
# Arranca a grabar tras N frames consecutivos con manos visibles, sin exigir movimiento de píxeles.
STATIC_HANDS_FRAMES_TO_START = 4
# Si el gesto tiene poco movimiento de landmarks, se interpreta como estático.
STATIC_GESTURE_MOTION_THRESHOLD = 0.012

# --- Corte de secuencia ---
STILL_FRAMES_LIMIT = 10
CAPTURE_BUFFER_SIZE = 60
MISSING_HANDS_LIMIT = 12
MIN_CAPTURE_FRAMES = 5

# Modos: "auto" (dinámico + estático), "dynamic", "static"
CAPTURE_MODE = "auto"

# ==========================================
# DATASET — RECOMENDACIONES PARA SEÑAS ESTÁTICAS
# ==========================================
# Clases que suelen ser poses cortas/fijas. Al grabar videos para estas clases:
#   1) Entrar en la pose → sostener 1–2 s → retirar manos.
#   2) Evitar movimientos largos de aproximación antes de la pose.
#   3) Preferir 50+ videos por clase (mismo criterio que train).
STATIC_SIGN_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "8",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M",
    "N", "ñ", "O", "P", "Q", "S", "T", "U", "V", "W", "X", "Y", "Z",
]