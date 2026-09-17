"""
Offsets y distancias mano–rostro sobre el vector (225,) ya normalizado.

No se guardan en el .npy: se calculan después de normalizar / espejar / aug,
porque mirror y augment_batch_3d asumen triples (x, y, z).

Y de MediaPipe crece hacia abajo. Un dy negativo = el índice está más arriba
que el punto de referencia (más cerca de ojos que de la boca, en general).
"""

from __future__ import annotations

import numpy as np

import config as cfg

POSE_NOSE = 0
POSE_LEFT_EYE = 2
POSE_RIGHT_EYE = 5
POSE_MOUTH_LEFT = 9
POSE_MOUTH_RIGHT = 10
HAND_WRIST = 0
HAND_INDEX_TIP = 8

FEATURE_NAMES = (
    "dy_index_mouth",
    "dy_index_eye",
    "dy_index_nose",
    "dist_index_mouth",
    "dist_index_eye",
    "dy_wrist_nose",
)

# Pose 0–10: cara gruesa que ya viene en Holistic (sin Face Mesh).
FACE_POSE_INDICES = (
    POSE_NOSE,
    1,
    POSE_LEFT_EYE,
    3,
    4,
    POSE_RIGHT_EYE,
    6,
    POSE_MOUTH_LEFT,
    POSE_MOUTH_RIGHT,
)


def _as_blocks(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.size < cfg.FRAME_FEATURES_DIM:
        raise ValueError(
            f"Se esperaban al menos {cfg.FRAME_FEATURES_DIM} valores, llegaron {vec.size}."
        )
    # Esta rama guarda pose + 2 manos. Si el vector es más largo, se ignoran extras.
    pose = vec[0 : cfg.POSE_DIM].reshape(33, 3)
    hands = vec[cfg.POSE_DIM : cfg.POSE_DIM + cfg.HANDS_DIM].reshape(2, 21, 3)
    return pose, hands[0], hands[1]


def _is_missing(points: np.ndarray) -> bool:
    return bool(np.all(points == 0.0))


def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def pick_active_hand(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray | None:
    """Tras el espejado zurdo la dominante queda en el bloque derecho."""
    if not _is_missing(right_hand):
        return right_hand
    if not _is_missing(left_hand):
        return left_hand
    return None


def closer_eye(index_tip: np.ndarray, pose: np.ndarray) -> np.ndarray:
    left_eye = pose[POSE_LEFT_EYE]
    right_eye = pose[POSE_RIGHT_EYE]
    if _xy_dist(index_tip, left_eye) <= _xy_dist(index_tip, right_eye):
        return left_eye
    return right_eye


def hand_face_features_frame(vector: np.ndarray) -> np.ndarray:
    """
    Un frame (225,) → (6,).

    Si falta la pose o las dos manos, devuelve ceros. No calcula distancia
    al origen: eso parecería “mano en el ancla de hombros”.
    """
    out = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    pose, left_hand, right_hand = _as_blocks(vector)
    if _is_missing(pose):
        return out
    hand = pick_active_hand(left_hand, right_hand)
    if hand is None:
        return out

    index_tip = hand[HAND_INDEX_TIP]
    wrist = hand[HAND_WRIST]
    if _is_missing(index_tip) and _is_missing(wrist):
        return out

    nose = pose[POSE_NOSE]
    mouth = 0.5 * (pose[POSE_MOUTH_LEFT] + pose[POSE_MOUTH_RIGHT])
    eye = closer_eye(index_tip, pose)
    tip = index_tip if not _is_missing(index_tip) else wrist

    out[0] = tip[1] - mouth[1]
    out[1] = tip[1] - eye[1]
    out[2] = tip[1] - nose[1]
    out[3] = _xy_dist(tip, mouth)
    out[4] = _xy_dist(tip, eye)
    out[5] = wrist[1] - nose[1]
    return out


def hand_face_features_sequence(sequence: np.ndarray) -> np.ndarray:
    """Secuencia (T, 225) → (T, 6)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Se esperaba (T, F), llegó shape={sequence.shape}")
    return np.stack(
        [hand_face_features_frame(sequence[t]) for t in range(sequence.shape[0])],
        axis=0,
    )


def visible_frame_mask(features: np.ndarray) -> np.ndarray:
    """True en frames donde alguna feature mano–rostro no es cero."""
    return ~np.all(features == 0.0, axis=1)


def clip_summary_row(features: np.ndarray) -> np.ndarray | None:
    """
    Un vector (6,) por video, para histogramas.

    Usa el frame visible con menor dist_index_eye (la mano más cerca del ojo).
    Si no hay mano, None.
    """
    mask = visible_frame_mask(features)
    if not np.any(mask):
        return None
    visible = features[mask]
    best = int(np.argmin(visible[:, FEATURE_NAMES.index("dist_index_eye")]))
    return visible[best]


def peak_frame_index(features: np.ndarray) -> int | None:
    mask = visible_frame_mask(features)
    if not np.any(mask):
        return None
    visible_idx = np.flatnonzero(mask)
    rel = int(np.argmin(features[mask, FEATURE_NAMES.index("dist_index_eye")]))
    return int(visible_idx[rel])


MOTION_NAMES = (
    "std_wrist_x",
    "std_wrist_y",
    "std_wrist_z",
    "range_wrist_x",
    "range_wrist_y",
    "range_wrist_z",
)


def clip_motion_stats(sequence: np.ndarray) -> np.ndarray | None:
    """
    Un vector (6,) por video: dispersion de la muñeca activa en x/y/z.

    Sirve para pares estatica vs dinamica (L/lunes, G/años, F/donde, V/viernes).
    V/viernes: viernes son 2 golpes al menton en Z con mano en V.
    None si hay menos de 3 frames con mano.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] < 3:
        return None
    points = []
    for t in range(sequence.shape[0]):
        _pose, left_hand, right_hand = _as_blocks(sequence[t])
        hand = pick_active_hand(left_hand, right_hand)
        if hand is None:
            continue
        wrist = hand[HAND_WRIST]
        if _is_missing(wrist):
            continue
        points.append(wrist)
    if len(points) < 3:
        return None
    arr = np.stack(points, axis=0)
    out = np.zeros(len(MOTION_NAMES), dtype=np.float32)
    out[0] = float(arr[:, 0].std())
    out[1] = float(arr[:, 1].std())
    out[2] = float(arr[:, 2].std())
    out[3] = float(arr[:, 0].max() - arr[:, 0].min())
    out[4] = float(arr[:, 1].max() - arr[:, 1].min())
    out[5] = float(arr[:, 2].max() - arr[:, 2].min())
    return out


# Configuración de dedos: extensión, curvatura y apertura.
# Las 6 features mano-rostro y las 6 de movimiento son ciegas a esto, y las
# confusiones más duras del dataset (D/B, U/K, M/N, 2/3, 4/5) son justamente
# de forma de mano con la mano en el mismo lugar.
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_MCPS = (2, 5, 9, 13, 17)
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
HAND_MIDDLE_MCP = 9

HANDSHAPE_NAMES = (
    "ext_thumb",
    "ext_index",
    "ext_middle",
    "ext_ring",
    "ext_pinky",
    "curl_thumb",
    "curl_index",
    "curl_middle",
    "curl_ring",
    "curl_pinky",
    "spread_thumb_index",
    "spread_index_middle",
    "spread_middle_ring",
    "spread_ring_pinky",
)


def handshape_features_frame(vector: np.ndarray) -> np.ndarray | None:
    """
    Un frame (225,) → (14,), invariante a escala y posición de la mano.

    Extensión = punta-muñeca / tamaño de mano.
    Curvatura = punta-nudillo / largo estirado del dedo (1 = recto, ~0 = cerrado).
    Apertura  = distancia entre puntas vecinas / tamaño de mano.
    None si no hay mano usable.
    """
    _pose, left_hand, right_hand = _as_blocks(vector)
    hand = pick_active_hand(left_hand, right_hand)
    if hand is None:
        return None
    wrist = hand[HAND_WRIST]
    scale = _xy_dist(wrist, hand[HAND_MIDDLE_MCP])
    if scale < 1e-6:
        return None

    out = np.zeros(len(HANDSHAPE_NAMES), dtype=np.float32)
    for i, tip in enumerate(FINGER_TIPS):
        out[i] = _xy_dist(hand[tip], wrist) / scale
    for i, (mcp, chain) in enumerate(zip(FINGER_MCPS, FINGER_CHAINS)):
        stretched = sum(
            _xy_dist(hand[chain[j]], hand[chain[j + 1]]) for j in range(len(chain) - 1)
        )
        out[5 + i] = _xy_dist(hand[chain[-1]], hand[mcp]) / stretched if stretched > 1e-6 else 0.0
    for i in range(len(FINGER_TIPS) - 1):
        out[10 + i] = _xy_dist(hand[FINGER_TIPS[i]], hand[FINGER_TIPS[i + 1]]) / scale
    return out


def clip_handshape_row(sequence: np.ndarray) -> np.ndarray | None:
    """Media de la configuración de dedos sobre los frames con mano."""
    sequence = np.asarray(sequence, dtype=np.float32)
    rows = [
        row
        for row in (handshape_features_frame(sequence[t]) for t in range(sequence.shape[0]))
        if row is not None
    ]
    if not rows:
        return None
    return np.stack(rows).mean(axis=0)


def face_and_hand_xy(vector: np.ndarray) -> dict:
    """Puntos 2D para el snapshot. Claves ausentes si falta pose o mano."""
    pose, left_hand, right_hand = _as_blocks(vector)
    if _is_missing(pose):
        return {}
    hand = pick_active_hand(left_hand, right_hand)
    nose = pose[POSE_NOSE]
    mouth = 0.5 * (pose[POSE_MOUTH_LEFT] + pose[POSE_MOUTH_RIGHT])
    data = {
        "nose": nose[:2].copy(),
        "mouth": mouth[:2].copy(),
        "left_eye": pose[POSE_LEFT_EYE, :2].copy(),
        "right_eye": pose[POSE_RIGHT_EYE, :2].copy(),
        "mouth_l": pose[POSE_MOUTH_LEFT, :2].copy(),
        "mouth_r": pose[POSE_MOUTH_RIGHT, :2].copy(),
        "shoulder_l": pose[11, :2].copy(),
        "shoulder_r": pose[12, :2].copy(),
    }
    if hand is None:
        return data
    data["index"] = hand[HAND_INDEX_TIP, :2].copy()
    data["wrist"] = hand[HAND_WRIST, :2].copy()
    data["eye"] = closer_eye(hand[HAND_INDEX_TIP], pose)[:2].copy()
    return data
