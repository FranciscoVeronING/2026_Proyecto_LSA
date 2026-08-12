"""Procesamiento de frames JPEG con MediaPipe Python (ruta de respaldo)."""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

_holistic = None


def _get_holistic():
    global _holistic
    if _holistic is None:
        _holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    return _holistic


def decode_jpeg_b64(image_b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def extract_raw_landmarks_from_frame(
    frame: np.ndarray, mirrored: bool = False
) -> tuple[list, list, list]:
    """Extrae landmarks crudos desde un frame BGR. Espeja si mirrored=True."""
    if mirrored:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic = _get_holistic()
    results = holistic.process(rgb)

    def _to_flat(landmarks, n_points: int) -> list:
        if not landmarks:
            return []
        return [
            coord
            for lm in landmarks.landmark
            for coord in (lm.x, lm.y, lm.z)
        ][: n_points * 3]

    pose = _to_flat(results.pose_landmarks, 33)
    left_hand = _to_flat(results.left_hand_landmarks, 21)
    right_hand = _to_flat(results.right_hand_landmarks, 21)
    return pose, left_hand, right_hand
