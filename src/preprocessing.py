import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Any
from tqdm import tqdm

from src.utils import get_anchor_and_scale, normalize_spatial_points, uniform_subsampling
from src.config import (
    DATASET_VIDEOS_DIR, 
    DATASET_NPY_DIR, 
    TARGET_FRAMES, 
    FRAME_FEATURES_DIM
)


def process_video_to_landmarks(
    video_path: str, 
    holistic_model: Any, 
    target_frames: int
) -> np.ndarray:
    """
    Abre un video, extrae los 543 landmarks de MediaPipe, filtra únicamente
    los 118 puntos requeridos por la arquitectura original y los normaliza.

    Returns:
        np.ndarray: Tensor de forma (target_frames, 708)
    """
    capture = cv2.VideoCapture(video_path)
    sequence_history: List[np.ndarray] = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(rgb_image)

        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) \
            if results.left_hand_landmarks else np.full((21, 3), np.nan)
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) \
            if results.pose_landmarks else np.full((33, 3), np.nan)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) \
            if results.right_hand_landmarks else np.full((21, 3), np.nan)

        # Matriz global
        all_landmarks = np.concatenate([lh, pose, rh], axis=0)

        # Reemplazar NaN por 0.0 si es necesario para evitar fallos numéricos en TensorFlow
        all_landmarks = np.nan_to_num(all_landmarks, nan=0.0)

        # Normalización espacial
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)
        
        raw_flat = all_landmarks.flatten()
        normalized_flat = normalize_spatial_points(raw_flat, anchor, scale)

        sequence_history.append(normalized_flat)

    capture.release()

    if not sequence_history:
        return np.zeros((target_frames, FRAME_FEATURES_DIM))

    # Submuestreo temporal uniforme
    return uniform_subsampling(sequence_history, target_frames=target_frames)


def run_extraction_pipeline(
    source_dir: str, 
    dest_dir: str,
    target_frames: int
) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    sign_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sign_class in sign_classes:
            print(f"\n[*] Processing category: {sign_class}")
            class_input_path = os.path.join(source_dir, sign_class)
            class_output_path = os.path.join(dest_dir, sign_class)
            os.makedirs(class_output_path, exist_ok=True)

            available_videos = glob.glob(os.path.join(class_input_path, "*.mp4"))

            for video_path in tqdm(available_videos, desc="Video Progress"):
                npy_filename = os.path.basename(video_path).replace('.mp4', '.npy')
                final_save_path = os.path.join(class_output_path, npy_filename)

                if os.path.exists(final_save_path):
                    continue

                try:
                    landmarks_tensor = process_video_to_landmarks(
                        video_path=video_path, 
                        holistic_model=holistic, 
                        target_frames=target_frames
                    )
                    if landmarks_tensor.shape[0] == target_frames:
                        np.save(final_save_path, landmarks_tensor)
                except Exception as process_error:
                    print(f"\n[!] Critical error in file {npy_filename}: {process_error}")


if __name__ == "__main__":
    EXPECTED_CHANNELS = FRAME_FEATURES_DIM
    print(f"Starting extraction pipeline:")
    print(f" > Expected features per frame: {EXPECTED_CHANNELS} channels")
    print(f" > Output temporal sequence: {TARGET_FRAMES} frames")
    
    run_extraction_pipeline(
        source_dir=DATASET_VIDEOS_DIR, 
        dest_dir=DATASET_NPY_DIR,
        target_frames=TARGET_FRAMES
    )