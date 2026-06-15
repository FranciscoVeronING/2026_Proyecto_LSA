import pandas as pd
import numpy as np
from typing import Tuple, List, Any

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
        return np.zeros((target_frames, 225), dtype=np.float32)

    # Generate equally spaced indices distributed throughout the footage
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    # Build the final matrix by indexing the original list
    filtered_sequence = [sequence_data[idx] for idx in indices]
    
    return np.array(filtered_sequence, dtype=np.float32)