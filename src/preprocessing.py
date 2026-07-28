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
    LIP, LHAND, RHAND, NOSE, REYE, LEYE, POINT_LANDMARKS, NUM_NODES
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

        # 1. Extraer los 543 landmarks exactamente en el orden del autor
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) \
            if results.face_landmarks else np.full((468, 3), np.nan)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) \
            if results.left_hand_landmarks else np.full((21, 3), np.nan)
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) \
            if results.pose_landmarks else np.full((33, 3), np.nan)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) \
            if results.right_hand_landmarks else np.full((21, 3), np.nan)

        # Matriz global (543, 3)
        all_landmarks = np.concatenate([face, lh, pose, rh], axis=0)

        # Reemplazar NaN por 0.0 si es necesario para evitar fallos numéricos en TensorFlow
        all_landmarks = np.nan_to_num(all_landmarks, nan=0.0)

        # 2. Filtrar únicamente los 118 puntos seleccionados
        selected_landmarks = all_landmarks[POINT_LANDMARKS]  # Shape: (118, 3)

        # 3. Normalización espacial
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)
        
        # Coordenadas brutas aplanadas (118 * 3 = 354)
        raw_flat = selected_landmarks.flatten()
        # Coordenadas normalizadas aplanadas (118 * 3 = 354)
        normalized_flat = normalize_spatial_points(raw_flat, anchor, scale)

        # Combinar brutas + normalizadas para obtener exactamente 708 características por frame
        frame_vector = np.concatenate([raw_flat, normalized_flat])  # Shape: (708,)

        sequence_history.append(frame_vector)

    capture.release()

    if not sequence_history:
        return np.zeros((target_frames, NUM_NODES * 6))

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
    EXPECTED_CHANNELS = NUM_NODES * 6  # 118 * 6 = 708
    print(f"Starting extraction pipeline:")
    print(f" > Selected Landmarks: {NUM_NODES} points")
    print(f" > Expected features per frame: {EXPECTED_CHANNELS} channels")
    print(f" > Output temporal sequence: {TARGET_FRAMES} frames")
    
    run_extraction_pipeline(
        source_dir=DATASET_VIDEOS_DIR, 
        dest_dir=DATASET_NPY_DIR,
        target_frames=TARGET_FRAMES
    )