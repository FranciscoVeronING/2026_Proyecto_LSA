
DATASET_VIDEOS_DIR = "../dataset"
DATASET_NPY_DIR = "../dataset_landmarks_32frames"
MODEL_SAVE_DIR = "../src/model"

NUM_CLASSES = 94
SAMPLES_PER_CLASS = 60

# Secuencia temporal unificada: preprocessing, entrenamiento e inferencia usan el mismo valor.
# Si tenés .npy viejos con otra cantidad de frames, train los re-muestrea automáticamente.
# Optuna v2 explora max_frames en {8, 12, 16, 24} subsampleando estos .npy (hace falta T>=24).
MAX_FRAMES = 10
TARGET_FRAMES = MAX_FRAMES


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
    "vivir_en"
]

USE_POSE = True
USE_HANDS = True
USE_FACE = False

POSE_DIM = 33 * 3 if USE_POSE else 0
HANDS_DIM = (21 * 3) * 2 if USE_HANDS else 0
FRAME_FEATURES_DIM = POSE_DIM + HANDS_DIM

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

SIGN_TO_INDEX = {sign: idx for idx, sign in enumerate(SIGN_CLASSES)}
INDEX_TO_SIGN = {idx: sign for idx, sign in enumerate(SIGN_CLASSES)}


# ==========================================
# HIPERPARÁMETROS DEL TINY TRANSFORMER
# ==========================================
HIDDEN_DIM = 256
NUM_HEADS = 2
NUM_LAYERS = 3
DROPOUT_RATE = 0.6

# ==========================================
# ENTRENAMIENTO Y DATA AUGMENTATION
# ==========================================
USE_DATA_AUGMENTATION = True
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 15
VIRTUAL_MULTIPLIER = 25
AUG_NOISE_STD = 0.026646077410572046
LR = 3.2356613600607546e-05
WEIGHT_DECAY = 0.014480175148028502
LABEL_SMOOTHING = 0.00025441433429255334
AUG_SCALE_RANGE = (0.85, 1.15)

# Ruido más fuerte en manos (tracker inestable) que en pose.
AUG_HAND_NOISE_STD = AUG_NOISE_STD
AUG_POSE_NOISE_STD = 0.012

# Rotación 3D pequeña (grados): yaw=persona de costado, pitch=cámara alta/baja, roll=inclinación.
AUG_ROT_YAW_DEG = 15.0
AUG_ROT_PITCH_DEG = 8.0
AUG_ROT_ROLL_DEG = 8.0

# Velocidad del gesto: >1 más rápido (menos frames), <1 más lento.
AUG_TIME_WARP_RANGE = (0.80, 1.25)

# Recorte temporal: fracción máxima a tirar de cada extremo antes de resamplear a MAX_FRAMES.
AUG_TEMPORAL_CROP_FRAC = 0.15

# Frames puestos a cero e interpolados (simula pérdidas de MediaPipe). 0–N por secuencia.
AUG_FRAME_DROPOUT_MAX = 3

# ==========================================
# INFERENCIA EN TIEMPO REAL (WEBCAM)
# ==========================================
CONFIDENCE_THRESHOLD = 0.75

# Segundos de espera tras una predicción antes de encolar otra inferencia
INFERENCE_COOLDOWN_SEC = 1.0

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
#   3) Preferir 60+ videos por clase (mismo criterio que train).
STATIC_SIGN_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "8",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M",
    "N", "ñ", "O", "P", "Q", "S", "T", "U", "V", "W", "X", "Y", "Z",
]