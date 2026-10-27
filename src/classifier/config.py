"""
Classifier and live-capture hyperparameters.

The extension asks for a subset via ``GET /config``. Changing ``MAX_FRAMES``
requires retraining: the ``.pth`` expects T=16.
"""

# ==========================================
# PATHS
# ==========================================
from pathlib import Path

# Absolute: the project must run from any working directory.
_CLASSIFIER_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _CLASSIFIER_DIR / "weights"

WEIGHTS_PATH = str(_WEIGHTS_DIR / "tinyskeleton_best.pth")
CLASSES_PATH = str(_WEIGHTS_DIR / "mapeo_clases.json")

MAX_FRAMES = 16

# Classifier labels: same order as ``mapeo_clases.json``.
# Not translator text; that goes through ``normalize_gloss``.
def _sign_classes_from_mapeo() -> list[str]:
    import json

    path = Path(CLASSES_PATH)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(data.items(), key=lambda kv: int(kv[1]))]


SIGN_CLASSES = _sign_classes_from_mapeo()

POSE_DIM = 33 * 3
HANDS_DIM = (21 * 3) * 2
FRAME_FEATURES_DIM = POSE_DIM + HANDS_DIM

# ==========================================
# TINY SKELETON CLASSIFIER
# ==========================================
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT_RATE = 0.4

# ==========================================
# REAL-TIME INFERENCE
# ==========================================
CONFIDENCE_THRESHOLD = 0.75

INFERENCE_COOLDOWN_SEC = 1.0 # Wait time after a prediction before accepting a new one

# Dynamic gesture detection thresholds
MOTION_PIXEL_THRESHOLD = 500
LANDMARK_MOTION_THRESHOLD = 0.008

# In Meet, Holistic often runs at 5–15 fps. Frame thresholds feel slow on a
# modest laptop: 6 ticks ≈ 1 s just to start recording.
STATIC_HANDS_FRAMES_TO_START = 4
STATIC_GESTURE_MOTION_THRESHOLD = 0.012

# Sign end detection thresholds
# Sign end from still hands (not the utterance pause toward the translator).
STILL_FRAMES_LIMIT = 10
CAPTURE_BUFFER_SIZE = 60
MISSING_HANDS_LIMIT = 12
HANDS_FRAMES_TO_START = 2
MIN_CAPTURE_FRAMES = 5

# Modes: "auto" (dynamic + static), "dynamic", "static"
CAPTURE_MODE = "auto"

# ==========================================
# BUFFER (glosses → translator)
# ==========================================
# Pause with no signing activity → close the list and send it to the translator.
# Counted from the last recognized sign or from when the hands stopped moving
# on camera, whichever happened last.
UTTERANCE_PAUSE_SEC = 4.0
# Consecutive equal letters allowed (the 3rd+ is dropped). Digits: no limit.
# Lexical signs ("other") are never accepted twice in a row.
LETTER_MAX_CONSECUTIVE = 2

# Classifier alias → token the translator sees.
GLOSS_NORMALIZER = {
    "el_ella": "EL",
    "esposo_a": "ESPOSO",
    "ahora_hoy": "HOY",
    "vivir_en": "VIVIR_EN",
    "hermano_a": "HERMANO",
    "hijo_a": "HIJO",
    "años": "AÑOS",
    "ñ": "Ñ",
}

STATIC_SIGN_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "8",
    "A", "B", "C", "D", "E", "F", "G", "I", "L", "M",
    "N", "ñ", "O", "P", "Q", "S", "T", "U", "V", "W", "X", "Y",
]