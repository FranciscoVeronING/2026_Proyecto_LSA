import numpy as np
from typing import Tuple, List, Any, Optional

from extraction.config import FRAME_FEATURES_DIM, MAX_FRAMES, MIN_CAPTURE_FRAMES, POSE_DIM, STATIC_GESTURE_MOTION_THRESHOLD

def get_anchor_and_scale(pose_landmarks: Any) -> Tuple[np.ndarray, float]:
    """Calcula ancla (punto medio de hombros) y escala inter-hombros."""
    if not pose_landmarks:
        return np.array([0.0, 0.0, 0.0]), 1.0

    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]

    anchor = np.array([
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
        (left_shoulder.z + right_shoulder.z) / 2.0,
    ])

    scale = float(np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    ))

    if scale < 1e-5:
        scale = 1.0

    return anchor, scale


def normalize_spatial_points(
    flat_landmarks: np.ndarray,
    anchor: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Traslada y escala un vector plano de landmarks."""
    if np.all(flat_landmarks == 0.0):
        return flat_landmarks

    points = flat_landmarks.reshape(-1, 3)
    normalized_points = (points - anchor) / scale
    return normalized_points.flatten()


def mirror_landmarks_for_left_handed(vector: np.ndarray, pose_dim: Optional[int] = None) -> np.ndarray:
    """
    Espeja landmarks normalizados para usuarios zurdos.
    Flip en X alrededor del origen corporal + swap bloques mano izq/der.
    """
    pose_dim = pose_dim if pose_dim is not None else POSE_DIM
    hand_dim = 21 * 3

    mirrored = vector.astype(np.float32).copy()
    points = mirrored.reshape(-1, 3)
    points[:, 0] *= -1.0
    mirrored = points.flatten()

    hand_start = pose_dim
    left_hand = mirrored[hand_start : hand_start + hand_dim].copy()
    right_hand = mirrored[hand_start + hand_dim : hand_start + 2 * hand_dim].copy()
    mirrored[hand_start : hand_start + hand_dim] = right_hand
    mirrored[hand_start + hand_dim : hand_start + 2 * hand_dim] = left_hand
    return mirrored


def interpolate_zero_frames(sequence: List[np.ndarray]) -> List[np.ndarray]:
    """
    Rellena frames donde MediaPipe devolvió ceros usando interpolación lineal
    por coordenada entre frames válidos consecutivos.
    """
    if len(sequence) < 2:
        return sequence

    arr = np.stack(sequence).astype(np.float32)
    zero_mask = np.all(arr == 0.0, axis=1)

    if not np.any(zero_mask) or np.all(zero_mask):
        return sequence

    valid_indices = np.where(~zero_mask)[0]
    if len(valid_indices) == 0:
        return sequence

    for dim in range(arr.shape[1]):
        arr[zero_mask, dim] = np.interp(
            np.where(zero_mask)[0],
            valid_indices,
            arr[valid_indices, dim],
        )

    return [arr[i] for i in range(len(arr))]


def normalize_sequence_to_frames(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """Lleva (T, F) a exactamente target_frames (subsampleo uniforme o padding)."""
    if sequence.size == 0:
        feature_dim = sequence.shape[1] if sequence.ndim == 2 else FRAME_FEATURES_DIM
        return np.zeros((target_frames, feature_dim), dtype=np.float32)

    frames_actuales, features = sequence.shape
    if frames_actuales == target_frames:
        return sequence.astype(np.float32)
    if frames_actuales < target_frames:
        padding = np.zeros((target_frames - frames_actuales, features), dtype=np.float32)
        return np.vstack((sequence, padding)).astype(np.float32)

    indices = np.linspace(0, frames_actuales - 1, target_frames, dtype=int)
    return sequence[indices].astype(np.float32)


def compute_landmark_hand_motion(
    current_vector: np.ndarray,
    previous_vector: Optional[np.ndarray],
    hand_start: int,
) -> float:
    """Movimiento L2 entre frames consecutivos, solo en la porción de manos."""
    if previous_vector is None:
        return 0.0
    curr_hands = current_vector[hand_start:]
    prev_hands = previous_vector[hand_start:]
    if np.all(curr_hands == 0.0) or np.all(prev_hands == 0.0):
        return 0.0
    return float(np.linalg.norm(curr_hands - prev_hands))


def trim_gesture_buffer(
    buffer: List[np.ndarray],
    hand_start: int,
    static_motion_threshold: float,
    min_frames: int = 5,
) -> List[np.ndarray]:
    """Recorta silencio al inicio/fin del buffer antes del subsampleo."""
    if len(buffer) <= min_frames:
        return buffer

    motions = [0.0]
    for idx in range(1, len(buffer)):
        motions.append(
            compute_landmark_hand_motion(buffer[idx], buffer[idx - 1], hand_start)
        )

    peak_motion = max(motions)
    if peak_motion < static_motion_threshold:
        keep = max(min_frames, len(buffer) // 2)
        return buffer[-keep:]

    active_threshold = peak_motion * 0.2
    active_indices = [i for i, motion in enumerate(motions) if motion >= active_threshold]
    if not active_indices:
        return buffer[-min_frames:]

    start = max(0, active_indices[0] - 2)
    end = min(len(buffer) - 1, active_indices[-1] + 4)
    trimmed = buffer[start : end + 1]
    return trimmed if len(trimmed) >= min_frames else buffer[-min_frames:]


def uniform_subsampling(sequence_data: List[np.ndarray], target_frames: int = 16) -> np.ndarray:
    """Comprime una secuencia variable a target_frames con índices equiespaciados."""
    total_frames = len(sequence_data)

    if total_frames == 0:
        feature_dim = len(sequence_data[0]) if sequence_data else FRAME_FEATURES_DIM
        return np.zeros((target_frames, feature_dim), dtype=np.float32)

    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    filtered_sequence = [sequence_data[idx] for idx in indices]
    return np.array(filtered_sequence, dtype=np.float32)


def sequence_buffer_to_model_input(
    buffer: List[np.ndarray],
    target_frames: Optional[int] = None,
    hand_start: Optional[int] = None,
    static_motion_threshold: Optional[float] = None,
    min_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Pipeline compartido cámara/preprocessing: trim → subsampleo uniforme.
    """
    target_frames = target_frames or MAX_FRAMES
    hand_start = hand_start if hand_start is not None else POSE_DIM
    static_motion_threshold = (
        static_motion_threshold or STATIC_GESTURE_MOTION_THRESHOLD
    )
    min_frames = min_frames or MIN_CAPTURE_FRAMES

    if len(buffer) < min_frames:
        return np.zeros((target_frames, FRAME_FEATURES_DIM), dtype=np.float32)

    trimmed = trim_gesture_buffer(
        buffer,
        hand_start=hand_start,
        static_motion_threshold=static_motion_threshold,
        min_frames=min_frames,
    )
    return uniform_subsampling(trimmed, target_frames=target_frames)

