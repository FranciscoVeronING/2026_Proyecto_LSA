"""Landmarks Holistic JS/WASM → vector 225-D y dibujo OpenCV (sin mediapipe Python)."""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from utils import get_anchor_and_scale, normalize_spatial_points

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def web_points_to_xyz(points: Any, expected: int) -> np.ndarray:
    """Lista JS [{x,y,z}, ...] → (expected, 3). Ausente = ceros."""
    if not points:
        return np.zeros((expected, 3), dtype=np.float32)
    arr = np.array(
        [[float(p["x"]), float(p["y"]), float(p.get("z", 0.0))] for p in points],
        dtype=np.float32,
    )
    if arr.shape[0] == expected:
        return arr
    out = np.zeros((expected, 3), dtype=np.float32)
    n = min(expected, arr.shape[0])
    out[:n] = arr[:n]
    return out


def extract_frame_vector(
    web_results: dict,
    use_pose: bool,
    use_hands: bool,
    use_face: bool,
) -> Optional[np.ndarray]:
    pose_xyz = web_points_to_xyz(web_results.get("pose"), 33)
    pose_present = bool(web_results.get("pose"))
    anchor, scale = get_anchor_and_scale(pose_xyz if pose_present else None)

    features_to_combine = []
    if use_pose:
        raw_pose = pose_xyz.flatten() if pose_present else np.zeros(33 * 3, dtype=np.float32)
        features_to_combine.append(normalize_spatial_points(raw_pose, anchor, scale))
    if use_face:
        face_xyz = web_points_to_xyz(web_results.get("face"), 468)
        raw_face = face_xyz.flatten() if web_results.get("face") else np.zeros(468 * 3, dtype=np.float32)
        features_to_combine.append(normalize_spatial_points(raw_face, anchor, scale))
    if use_hands:
        left_xyz = web_points_to_xyz(web_results.get("leftHand"), 21)
        right_xyz = web_points_to_xyz(web_results.get("rightHand"), 21)
        raw_left = left_xyz.flatten() if web_results.get("leftHand") else np.zeros(21 * 3, dtype=np.float32)
        raw_right = right_xyz.flatten() if web_results.get("rightHand") else np.zeros(21 * 3, dtype=np.float32)
        features_to_combine.append(normalize_spatial_points(raw_left, anchor, scale))
        features_to_combine.append(normalize_spatial_points(raw_right, anchor, scale))

    if not features_to_combine:
        return None
    return np.concatenate(features_to_combine)


def hands_present(web_results: dict) -> bool:
    return bool(web_results.get("leftHand") or web_results.get("rightHand"))


def _draw_points(image, points, connections, color, spec_color):
    h, w = image.shape[:2]
    xy = []
    for p in points:
        x = int(round(p["x"] * w))
        y = int(round(p["y"] * h))
        xy.append((x, y))
        cv2.circle(image, (x, y), 2, spec_color, -1, lineType=cv2.LINE_AA)
    for a, b in connections:
        if a < len(xy) and b < len(xy):
            cv2.line(image, xy[a], xy[b], color, 1, lineType=cv2.LINE_AA)


def draw_holistic(image, web_results: dict) -> None:
    pose = web_results.get("pose")
    if pose:
        _draw_points(image, pose, POSE_CONNECTIONS, (200, 180, 80), (0, 255, 255))
    left = web_results.get("leftHand")
    if left:
        _draw_points(image, left, HAND_CONNECTIONS, (0, 200, 0), (0, 255, 0))
    right = web_results.get("rightHand")
    if right:
        _draw_points(image, right, HAND_CONNECTIONS, (0, 120, 255), (0, 80, 255))
