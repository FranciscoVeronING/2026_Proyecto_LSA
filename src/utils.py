import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Optional

def get_anchor_and_scale(pose_landmarks: Any) -> Tuple[np.ndarray, float]:
    """
    Calculates the spatial anchor (midpoint of the chest) and the reference scale
    using the shoulder coordinates.

    Args:
        pose_landmarks: MediaPipe object containing pose landmarks.

    Returns:
        Tuple[np.ndarray, float]: A 3D vector (x, y, z) with the translation point
        and a float with the Euclidean scale factor.
    """
    if not pose_landmarks:
        return np.array([0.0, 0.0, 0.0]), 1.0

    # Standard MediaPipe Pose indices for shoulders
    # Extract nodes corresponding to the left and right shoulders
    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]

    # Translation: Exact midpoint between both shoulders (skeleton origin)
    anchor = np.array([
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
        (left_shoulder.z + right_shoulder.z) / 2.0
    ])

    # Scale: 2D Euclidean distance (X, Y) to mitigate depth distortions
    scale = float(np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    ))

    # Protect the pipeline against division by zero if MediaPipe fails critically
    if scale < 1e-5:
        scale = 1.0

    return anchor, scale



def normalize_spatial_points(
    flat_landmarks: np.ndarray, 
    anchor: np.ndarray, 
    scale: float
) -> np.ndarray:
    """
    Applies translation and scaling to a 1D vector of landmarks.

    Args:
        flat_landmarks (np.ndarray): 1D vector with sequential coordinates.
        anchor (np.ndarray): 1D vector (X, Y, Z) with the center of the chest.
        scale (float): Inter-shoulder scale divisor factor.

    Returns:
        np.ndarray: Normalized and flattened 1D vector.
    """
    # If the vector is all zeros, it means MediaPipe didn't detect the entity
    if np.all(flat_landmarks == 0.0):
        return flat_landmarks

    # Temporarily reshape to a 3D matrix (N, 3) for vector operations
    points = flat_landmarks.reshape(-1, 3)
    normalized_points = (points - anchor) / scale

    return normalized_points.flatten()


def normalize_sequence_to_frames(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Lleva una secuencia (T, F) a exactamente target_frames.
    Si hay más frames, subsamplea uniformemente (igual que en cámara).
    Si hay menos, rellena con ceros al final.
    """
    if sequence.size == 0:
        feature_dim = sequence.shape[1] if sequence.ndim == 2 else 225
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
    """
    Recorta silencio al inicio/fin del buffer antes del subsampleo.
    En gestos estáticos (poco movimiento), conserva los últimos frames (la pose sostenida).
    """
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
    """
    Temporally compresses the variable-length frame sequence 
    to a fixed size using uniformly spaced sampling.

    Args:
        sequence_data (List[np.ndarray]): List of arrays with processed frames.
        target_frames (int): Target number of frames for the TinyTransformer.

    Returns:
        np.ndarray: Dense matrix of shape (target_frames, 225).
    """
    total_frames = len(sequence_data)

    # Exception handling for empty or corrupt video files
    if total_frames == 0:
        feature_dim = len(sequence_data[0]) if sequence_data else 225
        return np.zeros((target_frames, feature_dim), dtype=np.float32)

    # Generate equally spaced indices distributed throughout the footage
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    # Build the final matrix by indexing the original list
    filtered_sequence = [sequence_data[idx] for idx in indices]
    
    return np.array(filtered_sequence, dtype=np.float32)