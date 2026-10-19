"""Une dos o más archivos .pose en un video de esqueleto con puentes.

Pipeline, en este orden:

1. Carga cada .pose y descarta POSE_WORLD_LANDMARKS (metros; el visor los
   dibujaría como píxeles y ensuciaría el origen).
2. Interpola a DEFAULT_FPS si el video original tenía otra cadencia.
3. Recorta una fracción al inicio y al final (preparación / retracción).
4. Rellena huecos cortos de Holistic. Manos fantasma / duplicadas / ausentes
   se estacionan al costado (si no, el puente deja la 2ª mano trabada).
5. EMA (media móvil exponencial) dentro de cada seña: reduce temblor.
6. Ancla hombros al centro de un canvas fijo (independiente de 720p/1080p).
7. Entre cada par: lerp con ease-in/out. La duración depende de las muñecas.
8. Dibuja con PoseVisualizer y escribe mp4 (OpenCV, codec mp4v, sin FFmpeg).

Ejemplos:

  py -3.11 extraction/render_pose_pair.py
  py -3.11 extraction/render_pose_pair.py --signs HOLA,MAMÁ,SALUDAR
  py -3.11 extraction/render_pose_pair.py --signs HOLA --spell FRANCISCO
  py -3.11 extraction/render_pose_pair.py --signs HOLA,F,R,A,N,C,I,S,C,O
  py -3.11 extraction/render_pose_pair.py --list-signs letras

O editar el bloque «PARÁMETROS EDITABLES» más abajo (son los defaults
de esos flags).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import numpy.ma as ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sign_names import desired_stem, strip_volume
POSE_DIR = ROOT / "out" / "poses"
OUT_DIR = ROOT / "out"
WINDOWS_PATH = OUT_DIR / "sign_windows.json"

# MediaPipe Pose: 15 = muñeca izquierda, 16 = muñeca derecha (índice 0-based).
WRIST_LEFT_IDX = 15
WRIST_RIGHT_IDX = 16
LEFT_ELBOW_IDX = 13
RIGHT_ELBOW_IDX = 14
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_PINKY_IDX = 17
RIGHT_PINKY_IDX = 18
LEFT_INDEX_TIP_IDX = 19
RIGHT_INDEX_TIP_IDX = 20
LEFT_THUMB_IDX = 21
RIGHT_THUMB_IDX = 22
HAND_WRIST = 0
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_PINKY_TIP = 20
# Rodillas / tobillos / talones / puntas: no se dibujan (plano cintura).
LEG_BELOW_HIP = slice(25, 33)
# Nariz/ojos/orejas/boca de pose: duplican FACE_LANDMARKS.
POSE_FACE = slice(0, 11)
# Meñique / índice / pulgar del cuerpo: duplican LEFT/RIGHT_HAND_LANDMARKS.
POSE_HAND_TIPS = slice(17, 23)
POSE_POINTS = 33
FACE_POINTS = 468
HAND_POINTS = 21
LEFT_HAND_START = POSE_POINTS + FACE_POINTS
RIGHT_HAND_START = LEFT_HAND_START + HAND_POINTS

# =============================================================================
# PARÁMETROS EDITABLES — continuidad intra-seña e inter-seña
# También se pueden pasar por CLI (mismos nombres, con --).
# =============================================================================

# Cadencia de salida. Si el .pose viene a otro fps, se interpola linealmente
# entre fotogramas (no alarga ni acorta la seña: solo cambia muestras/s).
DEFAULT_FPS = 30.0
# Velocidad de reproducción del mp4. 1.0 = tiempo real, 0.87 ≈ 15 % más lento.
DEFAULT_SPEED = 1.0

# EMA (Exponential Moving Average). Cada pose se mezcla con la anterior:
#   p_t = ALPHA * x_t + (1 - ALPHA) * p_{t-1}
# 1.00 = sin suavizado (sigue el landmark crudo, puede temblar).
# 0.85 = poco suavizado.
# 0.65 = default: recorta jitter sin “arrastrar” demasiado el movimiento.
# 0.40 = muy suave; el gesto se siente lento / retrasado.
DEFAULT_EMA = 0.50

# Recorte de bordes. "motion" usa out/sign_windows.json (inicio/fin por
# movimiento de manos, estilo VideoDivider). "fraction" corta el mismo %
# a ambos lados. "none" deja el clip entero.
DEFAULT_TRIM_MODE = "fraction"
# Fracción recortada al INICIO y al FINAL si trim-mode=fraction, o si no
# hay ventana de movimiento para esa seña (0.10 = 10 % + 10 %).
DEFAULT_TRIM = 0.25

# Duración del puente en frames. Si las muñecas casi no se desplazan entre
# el final de A y el inicio de B, se usa BRIDGE_MIN_FRAMES. Si se desplazan
# mucho, BRIDGE_MAX_FRAMES. En el medio se interpola linealmente.
# Más frames = transición más lenta / más “coarticulada”.
# Menos frames = corte más seco.
# Para un puente SIEMPRE igual de largo, poné el mismo valor en min y max
# o usá --bridge-fixed N.
BRIDGE_MIN_FRAMES = 4
BRIDGE_MAX_FRAMES = 8
# Distancia media de muñecas (píxeles) que mapea a min/max:
#   dist <= BRIDGE_DIST_SHORT_PX  →  BRIDGE_MIN_FRAMES
#   dist >= BRIDGE_DIST_LONG_PX   →  BRIDGE_MAX_FRAMES
# Bajar LONG hace que un desplazamiento chico ya pida el puente más largo.
# Subir SHORT hace que solo movimientos muy chicos usen el puente corto.
BRIDGE_DIST_SHORT_PX = 40.0
BRIDGE_DIST_LONG_PX = 180.0

# Curva del interpolador del puente:
#   "smoothstep" = ease-in/out (arranca y frena suave; default).
#   "linear"     = velocidad constante (se nota más el “desliz”).
BRIDGE_EASING = "smoothstep"

# Si no es None, ignora min/max y la distancia de muñecas: puente de N frames.
# Equivale a --bridge-fixed N. None = duración adaptativa.
BRIDGE_FIXED_FRAMES = None

# Rellenar landmarks que Holistic perdió (conf baja o x,y ≈ 0).
# Sin esto, el puente mezcla una mano real con (0,0) y se va a la esquina.
FILL_MISSING = True
MISSING_CONF_MIN = 0.30
MISSING_XY_PX = 8.0
# Huecos más largos que esto (frames) no se interpolan de punta a punta:
# si no, una mano que apareció 3 frames queda “trabada” todo el clip.
FILL_MAX_GAP = 8
# Seña a una mano: Holistic a veces duplica la hábil como izquierda, o el
# cuerpo inventa un brazo pasivo. Eso se estaciona al costado, no se puentea.
PARK_IDLE_HANDS = True
# Contacto (techo de CASA, etc.): una mano puede verse poco y NO es idle.
IDLE_HAND_HIT = 0.10
# Solo si las dos nubes de 21 puntos están en el mismo lugar (mano duplicada).
# 0.18 enganchaba Brazo/Cuantos: una mano sobre el otro antebrazo no es un duplicado.
DUPLICATE_HAND_FRAC = 0.07

# Canvas de salida: plano medio (cintura para arriba). Los .pose siguen en
# píxeles nativos; al unir se anclan los hombros y se recortan piernas y
# costados. 640×780 cabe en la mitad de un Meet 1920×1080.
CANVAS_WIDTH = 720
CANVAS_HEIGHT = 780
CANVAS_SHOULDER_PX = 250.0
# Fracción vertical del punto medio de hombros (0 = arriba, 1 = abajo).
# Más alto = más lugar para las manos; las caderas quedan cerca del borde.
CANVAS_SHOULDER_Y = 0.55
CENTER_ON_SHOULDERS = True
HIDE_LEGS = True
# No dibujar cara ni puntas de mano de POSE_LANDMARKS: se pisan con facemesh y hands.
HIDE_POSE_HAND_TIPS = True
HIDE_POSE_FACE = True

# =============================================================================


# Letras del abecedario LSA (incluye CH y Ñ).
# Tras el rename manual los archivos son `F.pose`, `HOLA.pose`, etc.
LETTER_ALIASES = {"ñ": "ñ", "n~": "ñ", "nh": "ñ"}
LETTER_TOKENS = set("abcdefghijklmnopqrstuvwxyz") | {"ch", "ñ"}


def fold_name(text: str) -> str:
    """HOLA / hola / MAMÁ / mama → misma clave ASCII minúscula."""
    stripped = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return stripped.lower().strip()


def fold_keep_enye(text: str) -> str:
    """Como fold_name pero conserva ñ (no la convierte en n)."""
    mapped = []
    for char in unicodedata.normalize("NFKD", text):
        if char.lower() == "ñ" or char == "\u0303":
            if char == "\u0303" and mapped and mapped[-1] == "n":
                mapped[-1] = "ñ"
            elif char.lower() == "ñ":
                mapped.append("ñ")
            continue
        if unicodedata.combining(char):
            continue
        mapped.append(char.lower())
    return "".join(mapped).strip()


def display_name(stem: str) -> str:
    return strip_volume(stem)


def list_pose_stems() -> list[str]:
    if not POSE_DIR.exists():
        return []
    return sorted(path.stem for path in POSE_DIR.glob("*.pose"))


def _tokens(stem: str) -> list[str]:
    name = fold_keep_enye(display_name(stem)).replace("_", " ")
    return [part for part in re.split(r"\s+", name) if part]


def _letter_query(raw: str) -> str | None:
    """'F' / 'ch' / 'Ñ' → token de letra, o None si no es dactilología."""
    key = fold_keep_enye(raw).replace(" ", "")
    key = LETTER_ALIASES.get(key, key)
    if key in LETTER_TOKENS:
        return key
    return None


def _alphabet_letter_tokens(stem: str) -> list[str] | None:
    """Si el archivo es una tarjeta del abecedario, las letras que tiene en el nombre."""
    letters: list[str] = []
    for part in _tokens(stem):
        if part == "abecedario":
            continue
        if part in LETTER_TOKENS:
            letters.append(part)
        else:
            return None
    return letters or None


def resolve_letter(letter: str, stems: list[str]) -> Path | None:
    """Una letra → el .pose del abecedario, no cualquier glosa que la contenga."""
    exact: list[str] = []
    first: list[str] = []
    other: list[str] = []
    for stem in stems:
        letters = _alphabet_letter_tokens(stem)
        if not letters:
            continue
        if letters == [letter]:
            exact.append(stem)
        elif letters[0] == letter:
            first.append(stem)
        elif letter in letters:
            other.append(stem)
    for group in (exact, first, other):
        if len(group) == 1:
            return POSE_DIR / f"{group[0]}.pose"
        if len(group) > 1:
            pick = group[0]
            return POSE_DIR / f"{pick}.pose"
    return None


def resolve_sign(name: str) -> Path:
    """Acepta `vol1_HOLA`, `HOLA`, `mama` o una letra del abecedario (`F`, `CH`)."""
    raw = name.strip()
    if not raw:
        raise FileNotFoundError("Nombre de seña vacío.")
    direct = POSE_DIR / f"{raw}.pose"
    if direct.exists():
        return direct
    canon = POSE_DIR / f"{desired_stem(raw)}.pose"
    if canon.exists():
        return canon

    stems = list_pose_stems()
    letter = _letter_query(raw)
    if letter:
        path = resolve_letter(letter, stems)
        if path:
            return path
        raise FileNotFoundError(
            f"No hay seña de dactilología para la letra {raw!r}. "
            "El archivo de la letra F es F.pose."
        )

    needle = fold_name(raw)
    exact = [stem for stem in stems if fold_name(stem) == needle]
    if not exact:
        exact = [stem for stem in stems if fold_name(display_name(stem)) == needle]
    if not exact and len(needle) >= 4:
        exact = [stem for stem in stems if needle in fold_name(stem)]
    if len(exact) == 1:
        return POSE_DIR / f"{exact[0]}.pose"
    if len(exact) > 1:
        preview = ", ".join(exact[:12])
        extra = f" (+{len(exact) - 12})" if len(exact) > 12 else ""
        raise FileNotFoundError(f"Varias coincidencias para {raw!r}: {preview}{extra}")
    raise FileNotFoundError(f"No hay .pose para {raw!r} en {POSE_DIR}. Probá --list-signs {raw}")


def expand_spell(text: str) -> list[str]:
    """FRANCISCO → ['F','R','A','N','C','I','S','C','O']. CH y Ñ cuentan como una seña."""
    compact = re.sub(r"[\s_\-]+", "", text)
    folded = fold_keep_enye(compact)
    letters: list[str] = []
    index = 0
    while index < len(folded):
        if folded[index : index + 2] == "ch":
            letters.append("CH")
            index += 2
            continue
        char = folded[index]
        if char == "ñ":
            letters.append("Ñ")
        elif char.isalpha():
            letters.append(char.upper())
        index += 1
    if not letters:
        raise ValueError(f"--spell {text!r} no tiene letras.")
    return letters


def label_for(requested: str, path: Path) -> str:
    """HUD: la letra pedida (F), no el nombre pegado del archivo (F G)."""
    if _letter_query(requested):
        letter = _letter_query(requested)
        return "Ñ" if letter == "ñ" else letter.upper()
    return display_name(path.stem)


def load_pose(path: Path) -> Pose:
    """Lee un .pose y se queda con pose/cara/manos en píxeles de imagen."""
    pose = Pose.read(path.read_bytes())
    # POSE_WORLD_LANDMARKS está en metros; el visor lo dibuja como píxeles y ensucia el origen.
    return pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    )


def interpolate(pose: Pose, fps: float) -> Pose:
    """Re-muestrea la seña a `fps` con lerp temporal. No cambia la duración en segundos."""
    if abs(float(pose.body.fps) - fps) < 0.5:
        return pose
    pose.body = pose.body.interpolate(new_fps=fps, kind="linear")
    return pose


def _valid_mask(data: np.ndarray, conf: np.ndarray, conf_min: float, xy_min: float) -> np.ndarray:
    """(T, P) True si el landmark tiene confianza y no está en el origen."""
    xy = np.linalg.norm(data[:, 0, :, :2], axis=-1)
    return (conf[:, 0, :] >= conf_min) & (xy >= xy_min)


def _fill_series(values: np.ndarray, ok: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Rellena huecos cortos con lerp; los largos se sostienen del vecino válido."""
    out = values.copy()
    filled = 0
    length = len(ok)
    if not ok.any() or ok.all():
        return out, 0
    index = 0
    while index < length:
        if ok[index]:
            index += 1
            continue
        start = index
        while index < length and not ok[index]:
            index += 1
        end = index
        left = start - 1
        right = end if end < length else -1
        gap = end - start
        if left >= 0 and right >= 0 and gap <= max_gap:
            xs = np.array([left, right], dtype=np.float32)
            out[start:end] = np.interp(np.arange(start, end, dtype=np.float32), xs, values[[left, right]])
            filled += gap
        elif left >= 0:
            out[start:end] = values[left]
            filled += gap
        elif right >= 0:
            out[start:end] = values[right]
            filled += gap
    return out, filled


def fill_missing_landmarks(
    pose: Pose,
    conf_min: float,
    xy_min: float,
    max_gap: int = FILL_MAX_GAP,
) -> tuple[Pose, int]:
    """Rellena huecos cortos de Holistic. No interpola de punta a punta el clip."""
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    valid = _valid_mask(data, conf, conf_min, xy_min)
    frames, _, points, dims = data.shape
    filled = 0
    for point in range(points):
        ok = valid[:, point]
        if ok.all() or not ok.any():
            continue
        conf[:, 0, point], _ = _fill_series(conf[:, 0, point], ok, max_gap)
        for axis in range(dims):
            data[:, 0, point, axis], n_fill = _fill_series(data[:, 0, point, axis], ok, max_gap)
        filled += n_fill
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    return pose, filled


def attach_empty_hands_to_wrists(pose: Pose, conf_min: float, xy_min: float) -> Pose:
    """Si una mano no apareció nunca en el clip, la pega a la muñeca del cuerpo."""
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if data.shape[2] < RIGHT_HAND_START + HAND_POINTS:
        return pose
    valid = _valid_mask(data, conf, conf_min, xy_min)
    for start, wrist in ((LEFT_HAND_START, WRIST_LEFT_IDX), (RIGHT_HAND_START, WRIST_RIGHT_IDX)):
        sl = slice(start, start + HAND_POINTS)
        if valid[:, sl].any():
            continue
        data[:, 0, sl, :] = data[:, 0, wrist : wrist + 1, :]
        conf[:, 0, sl] = np.maximum(conf[:, 0, wrist : wrist + 1], 0.35)
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    return pose


def _hand_frame_hit(valid: np.ndarray, start: int) -> float:
    return float(valid[:, start : start + HAND_POINTS].any(axis=1).mean())


def _hand_centroid(data: np.ndarray, start: int) -> np.ndarray:
    return data[:, 0, start : start + HAND_POINTS, :2].mean(axis=1)


def _shoulder_width_px(data: np.ndarray, conf: np.ndarray) -> float:
    widths = np.linalg.norm(
        data[:, 0, LEFT_SHOULDER_IDX, :2] - data[:, 0, RIGHT_SHOULDER_IDX, :2],
        axis=-1,
    )
    ok = (conf[:, 0, LEFT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        conf[:, 0, RIGHT_SHOULDER_IDX] >= MISSING_CONF_MIN
    )
    if ok.any():
        return float(max(np.median(widths[ok]), 80.0))
    return 400.0


def detect_idle_hands(
    pose: Pose,
    conf_min: float,
    xy_min: float,
    idle_hit: float = IDLE_HAND_HIT,
    dup_frac: float = DUPLICATE_HAND_FRAC,
) -> tuple[bool, bool]:
    """True = esa mano no está realmente en la seña (ausente, sparse o duplicada)."""
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if data.shape[2] < RIGHT_HAND_START + HAND_POINTS:
        return False, False
    valid = _valid_mask(data, conf, conf_min, xy_min)
    left_hit = _hand_frame_hit(valid, LEFT_HAND_START)
    right_hit = _hand_frame_hit(valid, RIGHT_HAND_START)
    idle_left = left_hit < idle_hit
    idle_right = right_hit < idle_hit
    both = valid[:, LEFT_HAND_START : LEFT_HAND_START + HAND_POINTS].any(axis=1) & valid[
        :, RIGHT_HAND_START : RIGHT_HAND_START + HAND_POINTS
    ].any(axis=1)
    if both.any() and not idle_left and not idle_right:
        left_c = _hand_centroid(data, LEFT_HAND_START)
        right_c = _hand_centroid(data, RIGHT_HAND_START)
        apart = float(np.median(np.linalg.norm(left_c[both] - right_c[both], axis=-1)))
        if apart < dup_frac * _shoulder_width_px(data, conf):
            if left_hit <= right_hit:
                idle_left = True
            else:
                idle_right = True
    if idle_left and idle_right:
        # Nunca estacionar las dos: nos quedamos con la de más presencia.
        if left_hit >= right_hit:
            idle_left = False
        else:
            idle_right = False
    return idle_left, idle_right


# Mano en reposo, dedos hacia +Y (abajo en imagen). Pulgar hacia −X (hacia el torso
# para la mano anatómica izquierda, que en cámara sale a la derecha del cuadro).
_REST_HAND_XY = np.array(
    [
        [0.00, 0.00],
        [-0.18, 0.10],
        [-0.24, 0.22],
        [-0.26, 0.32],
        [-0.28, 0.40],
        [-0.10, 0.14],
        [-0.10, 0.32],
        [-0.10, 0.44],
        [-0.10, 0.54],
        [0.00, 0.14],
        [0.00, 0.34],
        [0.00, 0.48],
        [0.00, 0.58],
        [0.10, 0.14],
        [0.10, 0.32],
        [0.10, 0.44],
        [0.10, 0.52],
        [0.18, 0.12],
        [0.20, 0.28],
        [0.20, 0.38],
        [0.20, 0.46],
    ],
    dtype=np.float32,
)


def _park_arm(
    data: np.ndarray,
    conf: np.ndarray,
    *,
    left: bool,
) -> None:
    """Brazo colgando al costado: codo, muñeca del cuerpo y 21 puntos de mano."""
    if left:
        shoulder_i, elbow_i, wrist_i = LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, WRIST_LEFT_IDX
        hip_i = LEFT_HIP_IDX
        hand_start = LEFT_HAND_START
        tips = (LEFT_PINKY_IDX, LEFT_INDEX_TIP_IDX, LEFT_THUMB_IDX)
        side = 1.0
        hand_xy = _REST_HAND_XY
    else:
        shoulder_i, elbow_i, wrist_i = RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, WRIST_RIGHT_IDX
        hip_i = RIGHT_HIP_IDX
        hand_start = RIGHT_HAND_START
        tips = (RIGHT_PINKY_IDX, RIGHT_INDEX_TIP_IDX, RIGHT_THUMB_IDX)
        side = -1.0
        hand_xy = _REST_HAND_XY.copy()
        hand_xy[:, 0] *= -1.0

    width = _shoulder_width_px(data, conf)
    shoulder = data[:, 0, shoulder_i]
    hip = data[:, 0, hip_i]
    hip_ok = (conf[:, 0, hip_i] >= 0.15) & (np.linalg.norm(hip[:, :2], axis=-1) >= MISSING_XY_PX)
    down = np.zeros_like(shoulder)
    down[:, 1] = width * 1.55
    if hip_ok.any():
        estimated = hip - shoulder
        estimated[~hip_ok] = down[~hip_ok]
        down = estimated
        down[:, 1] = np.maximum(down[:, 1], width * 0.9)
    outward = np.zeros_like(shoulder)
    outward[:, 0] = side * width * 0.12
    elbow = shoulder + 0.42 * down + 0.55 * outward
    wrist = shoulder + 0.90 * down + 0.70 * outward
    data[:, 0, elbow_i] = elbow
    data[:, 0, wrist_i] = wrist
    conf[:, 0, elbow_i] = np.maximum(conf[:, 0, elbow_i], 0.55)
    conf[:, 0, wrist_i] = np.maximum(conf[:, 0, wrist_i], 0.55)

    scale = width * 0.22
    offsets = np.zeros((HAND_POINTS, data.shape[-1]), dtype=np.float32)
    offsets[:, :2] = hand_xy * scale
    data[:, 0, hand_start : hand_start + HAND_POINTS] = wrist[:, None, :] + offsets[None, :, :]
    conf[:, 0, hand_start : hand_start + HAND_POINTS] = 0.55
    tip_src = (HAND_PINKY_TIP, HAND_INDEX_TIP, HAND_THUMB_TIP)
    for pose_idx, hand_idx in zip(tips, tip_src):
        data[:, 0, pose_idx] = data[:, 0, hand_start + hand_idx]
        conf[:, 0, pose_idx] = 0.55


def snap_pose_arm_to_hand(data: np.ndarray, conf: np.ndarray, *, left: bool) -> None:
    """La muñeca del cuerpo a menudo no coincide con la de la mano Holistic."""
    if left:
        shoulder_i, elbow_i, wrist_i = LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, WRIST_LEFT_IDX
        hand_start = LEFT_HAND_START
        hip_i = LEFT_HIP_IDX
        tips = (LEFT_PINKY_IDX, LEFT_INDEX_TIP_IDX, LEFT_THUMB_IDX)
    else:
        shoulder_i, elbow_i, wrist_i = RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, WRIST_RIGHT_IDX
        hand_start = RIGHT_HAND_START
        hip_i = RIGHT_HIP_IDX
        tips = (RIGHT_PINKY_IDX, RIGHT_INDEX_TIP_IDX, RIGHT_THUMB_IDX)
    hand_wrist = data[:, 0, hand_start + HAND_WRIST]
    data[:, 0, wrist_i] = hand_wrist
    conf[:, 0, wrist_i] = np.maximum(conf[:, 0, wrist_i], conf[:, 0, hand_start + HAND_WRIST])
    shoulder = data[:, 0, shoulder_i]
    hip = data[:, 0, hip_i]
    mid = 0.5 * (shoulder + hand_wrist)
    data[:, 0, elbow_i] = mid + 0.16 * (hip - shoulder)
    conf[:, 0, elbow_i] = np.maximum(conf[:, 0, elbow_i], 0.5)
    tip_src = (HAND_PINKY_TIP, HAND_INDEX_TIP, HAND_THUMB_TIP)
    for pose_idx, hand_idx in zip(tips, tip_src):
        data[:, 0, pose_idx] = data[:, 0, hand_start + hand_idx]
        conf[:, 0, pose_idx] = np.maximum(conf[:, 0, pose_idx], 0.5)


def ema_body(pose: Pose, alpha: float) -> Pose:
    """Suaviza cada landmark en el tiempo. alpha=1 deja el clip intacto."""
    if alpha >= 1.0 or pose.body.data.shape[0] < 2:
        return pose
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    out = np.empty_like(data)
    out[0] = data[0]
    keep = 1.0 - alpha
    for idx in range(1, len(data)):
        out[idx] = alpha * data[idx] + keep * out[idx - 1]
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(out), conf)
    return pose


def load_sign_windows() -> dict[str, dict]:
    if not WINDOWS_PATH.exists():
        return {}
    payload = json.loads(WINDOWS_PATH.read_text(encoding="utf-8"))
    return {item["stem"]: item for item in payload.get("items", [])}


_WINDOWS: dict[str, dict] | None = None


def sign_window(stem: str) -> dict | None:
    global _WINDOWS
    if _WINDOWS is None:
        _WINDOWS = load_sign_windows()
    return _WINDOWS.get(stem)


def slice_pose(pose: Pose, start: int, end: int) -> Pose:
    """Recorta un rango de frames [start, end) conservando header y fps."""
    data = pose.body.data[start:end]
    conf = pose.body.confidence[start:end]
    body = NumPyPoseBody(pose.body.fps, data, conf)
    return Pose(pose.header, body)


def trim_pose(pose: Pose, fraction: float) -> Pose:
    """Saca `fraction` al inicio y al final. Si el clip es corto, no recorta."""
    frames = pose.body.data.shape[0]
    if fraction <= 0 or frames < 12:
        return pose
    cut = max(1, int(round(frames * fraction)))
    start, end = cut, frames - cut
    if end - start < 8:
        return pose
    return slice_pose(pose, start, end)


def place_on_canvas(
    pose: Pose,
    width: int,
    height: int,
    shoulder_px: float,
    shoulder_y: float,
    hide_legs: bool = True,
) -> Pose:
    """Pasa de píxeles del video a un canvas fijo, hombros al centro.

    Traslación *por frame* (el torso no camina fuera del cuadro) y una sola
    escala por seña (mediana del ancho de hombros → `shoulder_px`). Así un
    clip mosaico y uno 1920×1080 quedan del mismo tamaño.
    """
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    left = data[:, 0, LEFT_SHOULDER_IDX, :]
    right = data[:, 0, RIGHT_SHOULDER_IDX, :]
    left_ok = (conf[:, 0, LEFT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        np.linalg.norm(left[:, :2], axis=-1) >= MISSING_XY_PX
    )
    right_ok = (conf[:, 0, RIGHT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        np.linalg.norm(right[:, :2], axis=-1) >= MISSING_XY_PX
    )
    ok = left_ok & right_ok
    mid = 0.5 * (left + right)
    widths = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
    times = np.arange(len(data), dtype=np.float32)
    if ok.any():
        for axis in range(data.shape[-1]):
            mid[:, axis] = np.interp(times, times[ok], mid[ok, axis])
        scale = float(shoulder_px) / max(float(np.median(widths[ok])), 1e-3)
    else:
        scale = 1.0
        mid[:] = 0.0
    cx = width / 2.0
    cy = height * float(shoulder_y)
    placed = data.copy()
    placed[..., 0] = (data[..., 0] - mid[:, None, None, 0]) * scale + cx
    placed[..., 1] = (data[..., 1] - mid[:, None, None, 1]) * scale + cy
    if data.shape[-1] > 2:
        placed[..., 2] = (data[..., 2] - mid[:, None, None, 2]) * scale
    if hide_legs and placed.shape[2] >= 33:
        conf = conf.copy()
        conf[:, 0, LEG_BELOW_HIP] = 0.0
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(placed), conf)
    pose.header.dimensions = PoseHeaderDimensions(width, height, pose.header.dimensions.depth)
    return pose


def prepare_sign(path: Path, args: argparse.Namespace) -> tuple[Pose, int, str]:
    """Carga → fps → trim → relleno corto → park/snap manos → EMA → canvas."""
    pose = interpolate(load_pose(path), args.fps)
    filled = 0
    parked = ""
    mode = getattr(args, "trim_mode", DEFAULT_TRIM_MODE)
    window = sign_window(path.stem) if mode == "motion" else None
    if window:
        fps = float(pose.body.fps or args.fps)
        start = int(round(float(window["start_sec"]) * fps))
        end = int(round(float(window["end_sec"]) * fps))
        frames = pose.body.data.shape[0]
        start = max(0, min(start, frames - 1))
        end = max(start + 1, min(end, frames))
        pose = slice_pose(pose, start, end)
    elif mode != "none":
        pose = trim_pose(pose, args.trim)
    idle_left, idle_right = detect_idle_hands(pose, args.missing_conf, args.missing_xy)
    if args.fill_missing:
        pose, filled = fill_missing_landmarks(
            pose,
            args.missing_conf,
            args.missing_xy,
            max_gap=getattr(args, "fill_max_gap", FILL_MAX_GAP),
        )
    park = getattr(args, "park_idle", PARK_IDLE_HANDS)
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if data.shape[2] >= RIGHT_HAND_START + HAND_POINTS:
        if park and idle_left:
            _park_arm(data, conf, left=True)
        elif not idle_left:
            snap_pose_arm_to_hand(data, conf, left=True)
        if park and idle_right:
            _park_arm(data, conf, left=False)
        elif not idle_right:
            snap_pose_arm_to_hand(data, conf, left=False)
        parked = f"{'L' if park and idle_left else ''}{'R' if park and idle_right else ''}"
        pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    pose = ema_body(pose, args.ema)
    if getattr(args, "center", CENTER_ON_SHOULDERS):
        pose = place_on_canvas(
            pose,
            args.canvas_width,
            args.canvas_height,
            args.shoulder_px,
            args.shoulder_y,
            hide_legs=getattr(args, "hide_legs", HIDE_LEGS),
        )
    return pose, filled, parked


def wrist_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Distancia media (píxeles) entre muñecas de dos poses (índices MediaPipe 15 y 16)."""
    left = float(np.linalg.norm(frame_a[WRIST_LEFT_IDX] - frame_b[WRIST_LEFT_IDX]))
    right = float(np.linalg.norm(frame_a[WRIST_RIGHT_IDX] - frame_b[WRIST_RIGHT_IDX]))
    return 0.5 * (left + right)


def ease_weights(t: np.ndarray, easing: str) -> np.ndarray:
    """t en [0,1] → pesos del interpolador. smoothstep frena en los extremos."""
    if easing == "linear":
        return t
    # Hermite cúbico: 3t² − 2t³. Derivada 0 en t=0 y t=1.
    return t * t * (3.0 - 2.0 * t)


def bridge_length(
    dist_px: float,
    min_len: int,
    max_len: int,
    dist_short: float,
    dist_long: float,
    fixed: int | None,
) -> int:
    """Elige cuántos frames dura el puente según la distancia de muñecas."""
    if fixed is not None and fixed > 0:
        return int(fixed)
    span = max(dist_long - dist_short, 1e-6)
    mix = float(np.clip((dist_px - dist_short) / span, 0.0, 1.0))
    return int(round(min_len + (max_len - min_len) * mix))


def bridge_poses(
    pose_a: Pose,
    pose_b: Pose,
    min_len: int,
    max_len: int,
    dist_short: float,
    dist_long: float,
    easing: str,
    fixed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lerp entre el último frame de A y el primero de B (cuerpo + confianza).

    No mezcla el interior de las señas: solo fabrica frames *nuevos* en el medio.
    """
    last = np.ma.filled(pose_a.body.data[-1], 0.0)
    first = np.ma.filled(pose_b.body.data[0], 0.0)
    last_c = np.ma.filled(pose_a.body.confidence[-1], 0.0)
    first_c = np.ma.filled(pose_b.body.confidence[0], 0.0)
    dist = wrist_distance(last[0], first[0])
    n = bridge_length(dist, min_len, max_len, dist_short, dist_long, fixed)
    n = max(n, 1)
    t = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
    w = ease_weights(t, easing).reshape(-1, 1, 1, 1)
    data = (1.0 - w) * last[None, ...] + w * first[None, ...]
    wc = ease_weights(t, easing).reshape(-1, 1, 1)
    conf = (1.0 - wc) * last_c[None, ...] + wc * first_c[None, ...]
    info = {"frames": n, "wrist_dist_px": round(dist, 1), "easing": easing}
    return data.astype(np.float32), conf.astype(np.float32), info


def compose_sequence(
    poses: list[Pose],
    names: list[str],
    args: argparse.Namespace,
) -> tuple[Pose, list[str], list[dict]]:
    """A + puente + B + puente + C + ..."""
    data_parts = [np.ma.filled(poses[0].body.data, 0.0)]
    conf_parts = [np.ma.filled(poses[0].body.confidence, 0.0)]
    labels = [ascii_label(names[0])] * poses[0].body.data.shape[0]
    bridges: list[dict] = []
    for index in range(1, len(poses)):
        prev_name = ascii_label(names[index - 1])
        this_name = ascii_label(names[index])
        bridge_data, bridge_conf, info = bridge_poses(
            poses[index - 1],
            poses[index],
            min_len=args.bridge_min,
            max_len=args.bridge_max,
            dist_short=args.bridge_dist_short,
            dist_long=args.bridge_dist_long,
            easing=args.easing,
            fixed=args.bridge_fixed,
        )
        info["from"] = prev_name
        info["to"] = this_name
        bridges.append(info)
        data_parts.append(bridge_data)
        conf_parts.append(bridge_conf)
        labels.extend([f"puente  {prev_name} -> {this_name}"] * bridge_data.shape[0])
        data_parts.append(np.ma.filled(poses[index].body.data, 0.0))
        conf_parts.append(np.ma.filled(poses[index].body.confidence, 0.0))
        labels.extend([this_name] * poses[index].body.data.shape[0])
    body = NumPyPoseBody(
        poses[0].body.fps,
        ma.array(np.concatenate(data_parts, axis=0)),
        np.concatenate(conf_parts, axis=0),
    )
    return Pose(poses[0].header, body), labels, bridges


def hide_pose_overlays(pose: Pose, hide_hand_tips: bool = True, hide_face: bool = True) -> Pose:
    """Apaga cara y puntas de mano del cuerpo para no tapar facemesh ni las 21 de cada mano."""
    n_points = pose.body.confidence.shape[-1]
    if n_points < 11:
        return pose
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32).copy()
    if hide_face:
        conf[:, 0, POSE_FACE] = 0.0
    if hide_hand_tips and n_points >= 23:
        conf[:, 0, POSE_HAND_TIPS] = 0.0
    pose.body = NumPyPoseBody(pose.body.fps, pose.body.data, conf)
    return pose


def write_video(
    pose: Pose,
    labels: list[str],
    path: Path,
    hide_hand_tips: bool = True,
    hide_pose_face: bool = True,
    speed: float = 1.0,
) -> None:
    """Dibuja cada frame (esqueleto + HUD de fase) y lo escribe a mp4v."""
    import cv2
    from pose_format.pose_visualizer import PoseVisualizer

    if hide_hand_tips or hide_pose_face:
        pose = hide_pose_overlays(pose, hide_hand_tips=hide_hand_tips, hide_face=hide_pose_face)
    play_fps = max(float(pose.body.fps) * float(speed), 1.0)
    visualizer = PoseVisualizer(pose, thickness=2)
    writer = None
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(visualizer.draw(background_color=(18, 18, 22))):
        text = labels[index]
        color = (60, 180, 255) if text.startswith("puente") else (80, 220, 120)
        cv2.putText(frame, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{index + 1}/{len(labels)}",
            (24, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), play_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"No se pudo abrir VideoWriter en {path}")
        writer.write(frame)
    if writer is not None:
        writer.release()


def ascii_label(text: str) -> str:
    """Quita acentos (OpenCV Hershey no dibuja Á). MAMÁ → MAMA."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def parse_sign_list(args: argparse.Namespace) -> list[str]:
    items: list[str] = []
    if args.signs.strip():
        items = [item.strip() for item in args.signs.split(",") if item.strip()]
    elif not args.spell.strip():
        items = [args.sign_a, args.sign_b]
    if args.spell.strip():
        items.extend(expand_spell(args.spell))
    if len(items) < 2:
        raise ValueError(
            "Hacen falta al menos dos señas. Ejemplos:\n"
            "  --signs HOLA,MAMÁ,SALUDAR\n"
            "  --signs HOLA --spell FRANCISCO\n"
            "  --signs HOLA,F,R,A,N,C,I,S,C,O"
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Une dos o más .pose en un mp4. El puente se regula con --bridge-* / --trim / --ema."
    )
    parser.add_argument(
        "--signs",
        default="",
        help="Lista de glosas o letras separadas por coma (HOLA,F,R,A). Si se omite, usa --sign-a/--sign-b.",
    )
    parser.add_argument(
        "--spell",
        default="",
        help="Dactilología: FRANCISCO se expande a F,R,A,N,C,I,S,C,O (CH y Ñ son una seña).",
    )
    parser.add_argument("--sign-a", default="HOLA", help="Primera seña si no pasás --signs.")
    parser.add_argument("--sign-b", default="MAMÁ", help="Segunda seña si no pasás --signs.")
    parser.add_argument(
        "--list-signs",
        nargs="?",
        const="*",
        default="",
        help="Lista stems en out/poses/. Filtro opcional, o 'letras' para el abecedario.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Cadencia de salida (default: %(default)s).")
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Velocidad del mp4: 1=tiempo real, 0.87≈15%% más lento (default: %(default)s).",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=DEFAULT_EMA,
        help="Suavizado intra-seña 0–1. 1=crudo, 0.4=muy suave (default: %(default)s).",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=DEFAULT_TRIM,
        help="Fracción recortada al inicio y al final si --trim-mode fraction (default: %(default)s).",
    )
    parser.add_argument(
        "--trim-mode",
        choices=("motion", "fraction", "none"),
        default=DEFAULT_TRIM_MODE,
        help="motion=ventanas de sign_windows.json; fraction=%% fijo; none=sin recorte.",
    )
    parser.add_argument(
        "--bridge-min",
        type=int,
        default=BRIDGE_MIN_FRAMES,
        help="Frames del puente si las muñecas casi no se mueven (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge-max",
        type=int,
        default=BRIDGE_MAX_FRAMES,
        help="Frames del puente si las muñecas se desplazan mucho (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge-dist-short",
        type=float,
        default=BRIDGE_DIST_SHORT_PX,
        help="Distancia de muñecas (px) que mapea a --bridge-min (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge-dist-long",
        type=float,
        default=BRIDGE_DIST_LONG_PX,
        help="Distancia de muñecas (px) que mapea a --bridge-max (default: %(default)s).",
    )
    parser.add_argument(
        "--easing",
        choices=("smoothstep", "linear"),
        default=BRIDGE_EASING,
        help="Curva del interpolador del puente (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge-fixed",
        type=int,
        default=BRIDGE_FIXED_FRAMES,
        help="Si se pasa, cada puente dura exactamente N frames (ignora min/max y muñecas).",
    )
    fill = parser.add_mutually_exclusive_group()
    fill.add_argument(
        "--fill-missing",
        dest="fill_missing",
        action="store_true",
        default=FILL_MISSING,
        help="Rellena manos/puntos en (0,0) antes del puente (default: activo).",
    )
    fill.add_argument(
        "--no-fill-missing",
        dest="fill_missing",
        action="store_false",
        help="Deja los ceros de Holistic: las manos pueden volar a la esquina en el puente.",
    )
    parser.add_argument("--missing-conf", type=float, default=MISSING_CONF_MIN)
    parser.add_argument("--missing-xy", type=float, default=MISSING_XY_PX)
    parser.add_argument(
        "--fill-max-gap",
        type=int,
        default=FILL_MAX_GAP,
        help="Huecos más largos no se interpolan (default: %(default)s frames).",
    )
    park = parser.add_mutually_exclusive_group()
    park.add_argument(
        "--park-idle",
        dest="park_idle",
        action="store_true",
        default=PARK_IDLE_HANDS,
        help="Estaciona la mano que Holistic no vio / duplicó (default: activo).",
    )
    park.add_argument(
        "--no-park-idle",
        dest="park_idle",
        action="store_false",
        help="Deja la 2ª mano como vino: puede quedar trabada entre señas.",
    )
    center = parser.add_mutually_exclusive_group()
    center.add_argument(
        "--center",
        dest="center",
        action="store_true",
        default=CENTER_ON_SHOULDERS,
        help="Ancla hombros al centro del canvas (default: activo).",
    )
    center.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="Deja coordenadas nativas: mosaico y 1080p no coinciden.",
    )
    parser.add_argument("--canvas-width", type=int, default=CANVAS_WIDTH)
    parser.add_argument("--canvas-height", type=int, default=CANVAS_HEIGHT)
    parser.add_argument(
        "--shoulder-px",
        type=float,
        default=CANVAS_SHOULDER_PX,
        help="Ancho de hombros en el canvas (default: %(default)s).",
    )
    parser.add_argument(
        "--shoulder-y",
        type=float,
        default=CANVAS_SHOULDER_Y,
        help="Altura relativa del medio de hombros, 0=arriba (default: %(default)s).",
    )
    legs = parser.add_mutually_exclusive_group()
    legs.add_argument(
        "--hide-legs",
        dest="hide_legs",
        action="store_true",
        default=HIDE_LEGS,
        help="No dibuja de la rodilla para abajo (default: activo).",
    )
    legs.add_argument(
        "--show-legs",
        dest="hide_legs",
        action="store_false",
        help="Dibuja el esqueleto completo.",
    )
    tips = parser.add_mutually_exclusive_group()
    tips.add_argument(
        "--hide-pose-hands",
        dest="hide_pose_hands",
        action="store_true",
        default=HIDE_POSE_HAND_TIPS,
        help="No dibuja meñique/índice/pulgar de pose (default: activo).",
    )
    tips.add_argument(
        "--show-pose-hands",
        dest="hide_pose_hands",
        action="store_false",
        help="Dibuja también los 3 puntos de mano del cuerpo.",
    )
    face = parser.add_mutually_exclusive_group()
    face.add_argument(
        "--hide-pose-face",
        dest="hide_pose_face",
        action="store_true",
        default=HIDE_POSE_FACE,
        help="No dibuja nariz/ojos/boca de pose; queda el facemesh (default: activo).",
    )
    face.add_argument(
        "--show-pose-face",
        dest="hide_pose_face",
        action="store_false",
        help="Dibuja también los 11 puntos de cara del cuerpo.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    return parser.parse_args()


def print_sign_list(filter_text: str) -> None:
    stems = list_pose_stems()
    if not stems:
        print(f"No hay .pose en {POSE_DIR}")
        return
    if fold_name(filter_text) in {"letra", "letras", "abecedario", "alfabeto"}:
        print("Dactilologia (letra -> archivo). Varias quedaron pegadas con el vecino en el PDF:")
        for letter in list("abcdefghijklmnopqrstuvwxyz") + ["ch", "ñ"]:
            path = resolve_letter(letter, stems)
            shown = letter.upper() if letter != "ñ" else "Ñ"
            if letter == "ch":
                shown = "CH"
            print(f"  {shown:>3}  {path.stem if path else '(no está)'}")
        return
    needle = "" if filter_text in {"", "*"} else fold_name(filter_text)
    shown = [stem for stem in stems if needle in fold_name(stem)]
    print(f"{len(shown)} señas" + (f" con {filter_text!r}" if needle else f" en {POSE_DIR}"))
    for stem in shown:
        print(stem)


def save_previews(video_path: Path, labels: list[str], preview: Path) -> None:
    import cv2

    preview.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    picks = {0, max(len(labels) - 1, 0)}
    seen: set[str] = set()
    for index, label in enumerate(labels):
        if label not in seen:
            picks.add(index)
            seen.add(label)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in picks:
            cv2.imwrite(str(preview / f"frame_{idx:04d}.png"), frame)
        idx += 1
    cap.release()


def main() -> None:
    args = parse_args()
    if args.list_signs:
        print_sign_list(args.list_signs)
        return

    requested = parse_sign_list(args)
    paths = [resolve_sign(name) for name in requested]
    names = [label_for(name, path) for name, path in zip(requested, paths)]
    for requested_name, path, name in zip(requested, paths, names):
        if _letter_query(requested_name) and display_name(path.stem) != name:
            print(f"{name} ← {path.stem}")

    prepared: list[Pose] = []
    for path, name in zip(paths, names):
        pose, filled, parked = prepare_sign(path, args)
        prepared.append(pose)
        extra = (
            f", canvas={pose.header.dimensions.width}x{pose.header.dimensions.height}"
            if args.center
            else ""
        )
        park_bit = f", park={parked}" if parked else ""
        print(
            f"{name}: {pose.body.data.shape[0]} frames "
            f"(trim={args.trim_mode}, ema={args.ema:g}, rellenos={filled}{park_bit}{extra})"
        )

    clip, labels, bridges = compose_sequence(prepared, names, args)
    slug = "_".join(ascii_label(name).replace(" ", "_") for name in names)
    output = args.output or (OUT_DIR / f"{slug}.mp4")
    speed = float(getattr(args, "speed", DEFAULT_SPEED) or 1.0)
    write_video(
        clip,
        labels,
        output,
        hide_hand_tips=getattr(args, "hide_pose_hands", HIDE_POSE_HAND_TIPS),
        hide_pose_face=getattr(args, "hide_pose_face", HIDE_POSE_FACE),
        speed=speed,
    )
    preview = args.preview_dir or (OUT_DIR / f"preview_{slug.lower()}")
    save_previews(output, labels, preview)

    for info in bridges:
        print(
            f"Puente {info['from']} -> {info['to']}: {info['frames']} frames  "
            f"(muñecas {info['wrist_dist_px']} px, easing={info['easing']})"
        )
    play_fps = max(float(clip.body.fps) * speed, 1.0)
    print(
        f"Total: {len(labels)} @ {play_fps:g} fps "
        f"({len(labels) / play_fps:.2f} s, speed={speed:g})"
    )
    print(f"Video: {output}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
