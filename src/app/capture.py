"""Captura de video y armado del vector de landmarks que consume el clasificador."""

from threading import Thread

import cv2
import numpy as np
import torch

import classifier.config as cfg
from core.landmarks import (
    normalize_spatial_points,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)


class WebcamStream:
    """Lectura de la webcam en un hilo aparte para no frenar el loop de dibujo."""

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            print("[!] ERROR CRITICO: no se puede abrir la camara.")
            self.stopped = True
            return

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
            else:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class LandmarkSmoother:
    """Media exponencial sobre el vector de landmarks (atenúa el jitter de MediaPipe)."""

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

    def reset(self):
        self.prev_vector = None


def _flat_to_xyz(flat: list, expected_points: int) -> np.ndarray:
    """Lista plana [x,y,z,...] → array (N*3,) con padding de ceros si falta."""
    if not flat:
        return np.zeros(expected_points * 3, dtype=np.float32)
    arr = np.array(flat, dtype=np.float32)
    target = expected_points * 3
    if arr.size < target:
        padded = np.zeros(target, dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr[:target]


def _anchor_scale_from_flat_pose(pose_flat: list) -> tuple[np.ndarray, float]:
    """Calcula ancla y escala desde hombros (índices 11 y 12) en lista plana."""
    if not pose_flat or len(pose_flat) < 12 * 3:
        return np.array([0.0, 0.0, 0.0]), 1.0

    pose = np.array(pose_flat, dtype=np.float32).reshape(-1, 3)
    if pose.shape[0] < 13:
        return np.array([0.0, 0.0, 0.0]), 1.0

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    anchor = (left_shoulder + right_shoulder) / 2.0
    scale = float(np.sqrt(np.sum((left_shoulder - right_shoulder) ** 2)))
    if scale < 1e-5:
        scale = 1.0
    return anchor, scale


def extract_normalized_vector_from_flat(
    pose_flat: list,
    left_hand_flat: list,
    right_hand_flat: list,
    left_handed: bool = False,
) -> np.ndarray:
    """Arma el vector (225,) desde listas planas de landmarks crudos."""
    anchor, scale = _anchor_scale_from_flat_pose(pose_flat)
    raw_pose = _flat_to_xyz(pose_flat, 33)
    raw_lh = _flat_to_xyz(left_hand_flat, 21)
    raw_rh = _flat_to_xyz(right_hand_flat, 21)

    norm_pose = normalize_spatial_points(raw_pose, anchor, scale)
    norm_lh = normalize_spatial_points(raw_lh, anchor, scale)
    norm_rh = normalize_spatial_points(raw_rh, anchor, scale)
    vector = np.concatenate([norm_pose, norm_lh, norm_rh])

    if left_handed:
        vector = mirror_landmarks_for_left_handed(vector, pose_dim=cfg.POSE_DIM)
    return vector


def extract_normalized_vector(results, left_handed: bool):
    """Arma el vector (225,) desde los resultados de MediaPipe; espeja si es zurdo."""
    raw_pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else []
    )
    raw_lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else []
    )
    raw_rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else []
    )
    return extract_normalized_vector_from_flat(raw_pose, raw_lh, raw_rh, left_handed=left_handed)


def prepare_input_tensor(buffer_list, device):
    """Lista de frames → tensor (1, MAX_FRAMES, features). None si no alcanza."""
    matrix = sequence_buffer_to_model_input(buffer_list)
    if matrix.shape[0] != cfg.MAX_FRAMES:
        return None
    tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)


def should_start_recording(
    capture_mode,
    hands_present,
    is_moving,
    consecutive_hands_frames,
    static_frames_to_start=None,
):
    static_frames = static_frames_to_start or cfg.STATIC_HANDS_FRAMES_TO_START
    if not hands_present:
        return False
    if capture_mode == "dynamic":
        return is_moving
    if capture_mode == "static":
        return consecutive_hands_frames >= static_frames
    static_ready = consecutive_hands_frames >= static_frames
    return is_moving or static_ready
