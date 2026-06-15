
DATASET_VIDEOS_DIR = "./dataset"
DATASET_NPY_DIR = "./dataset_landmarks"
MODEL_SAVE_DIR = "/src/model"

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
    "el/ella"
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
    "hora"
    "familia",
    "mama",
    "papa",
    "hermano/a",
    "tener",
    #"arma",
    "cuchillo",
    "brazo",
    "cara",
    "hijo/a",
    "numero",
    "años(Edad)",
    "ojo",
    "esposo/a",
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
    "hoy/ahora",
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
    "vivir en"
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

MAX_FRAMES = 16
BATCH_SIZE = 32
EPOCHS_BASE = 15
LEARNING_RATE_TRANSFER = 1e-3
LEARNING_RATE_FINE_TUNING = 1e-5
PATIENCE = 15