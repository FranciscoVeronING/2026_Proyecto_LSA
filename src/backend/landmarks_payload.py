"""Convierte landmarks JSON de la extensión al vector 225 del clasificador."""

from types import SimpleNamespace

import numpy as np

import classifier.config as cfg
from core.landmarks import (
    get_anchor_and_scale,
    mirror_landmarks_for_left_handed,
    normalize_spatial_points,
    sequence_buffer_to_model_input,
)


class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_vector = None

    def update(self, new_vector):
        if self.prev_vector is None:
            self.prev_vector = new_vector
            return new_vector
        smoothed = (self.alpha * new_vector) + ((1 - self.alpha) * self.prev_vector)
        self.prev_vector = smoothed
        return smoothed


def _as_xyz(point):
    if isinstance(point, dict):
        return (float(point["x"]), float(point["y"]), float(point.get("z") or 0.0))
    return (float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0))


def _landmark_bag(points):
    if not points:
        return None
    lms = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in (_as_xyz(p) for p in points)]
    return SimpleNamespace(landmark=lms)


def _flat_or_zeros(bag, n_points: int) -> np.ndarray:
    if not bag:
        return np.zeros(n_points * 3)
    return np.array([[lm.x, lm.y, lm.z] for lm in bag.landmark], dtype=np.float32).flatten()


def vector_from_frame(frame: dict, left_handed: bool) -> np.ndarray:
    """Un frame {pose, left_hand, right_hand} → vector (225,) normalizado."""
    pose = _landmark_bag(frame.get("pose"))
    left_hand = _landmark_bag(frame.get("left_hand"))
    right_hand = _landmark_bag(frame.get("right_hand"))
    anchor, scale = get_anchor_and_scale(pose)
    vector = np.concatenate(
        [
            normalize_spatial_points(_flat_or_zeros(pose, 33), anchor, scale),
            normalize_spatial_points(_flat_or_zeros(left_hand, 21), anchor, scale),
            normalize_spatial_points(_flat_or_zeros(right_hand, 21), anchor, scale),
        ]
    )
    if left_handed:
        vector = mirror_landmarks_for_left_handed(vector, pose_dim=cfg.POSE_DIM)
    return vector


def frames_to_matrix(vectors):
    return sequence_buffer_to_model_input(vectors)
