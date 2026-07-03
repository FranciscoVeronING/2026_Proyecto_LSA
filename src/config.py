
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_VIDEOS_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_NPY_DIR = os.path.join(BASE_DIR, "dataset_landmarks")
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

USE_POSE = True
USE_HANDS = True
USE_FACE = True

POSE_DIM = 33 * 3 if USE_POSE else 0
HANDS_DIM = (21 * 3) * 2 if USE_HANDS else 0
FACE_DIM = 468 * 3 if USE_FACE else 0
FRAME_FEATURES_DIM = POSE_DIM + HANDS_DIM + FACE_DIM

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FRAMES = 16

SIGN_TO_INDEX = {sign: idx for idx, sign in enumerate(SIGN_CLASSES)}
INDEX_TO_SIGN = {idx: sign for idx, sign in enumerate(SIGN_CLASSES)}

BATCH_SIZE = 32
EPOCHS_BASE = 15
EPOCHS = 25
LEARNING_RATE_TRANSFER = 1e-3
LEARNING_RATE_FINE_TUNING = 1e-5
PATIENCE = 15
VAL_SIZE = 0.2