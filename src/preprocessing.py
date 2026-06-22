import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Any
from tqdm import tqdm

from utils import get_anchor_and_scale, normalize_spatial_points, uniform_subsampling
from config import (
    DATASET_VIDEOS_DIR, 
    DATASET_NPY_DIR, 
    TARGET_FRAMES,
    USE_FACE, 
    USE_POSE, 
    USE_HANDS,
    FRAME_FEATURES_DIM
)

def process_video_to_landmarks(
    video_path: str, 
    holistic_model: Any, 
    target_frames: int,
    use_pose: bool,
    use_hands: bool,
    use_face: bool
) -> np.ndarray:
    """
    Opens a video file, extracts, spatially normalizes each frame, 
    and applies uniform subsampling to the resulting vector.

    Args:
        video_path (str): Path to the MP4 file.
        holistic_model (Any): Contextual instance of MediaPipe Holistic.
        target_frames (int): Exact number of output frames.

    Returns:
        np.ndarray: Structured tensor of shape (target_frames, 225).
    """
    capture = cv2.VideoCapture(video_path)
    sequence_history: List[np.ndarray] = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Mandatory color space conversion for MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(rgb_image)

        # Determine the local coordinate system for this specific frame
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)

        features_to_combine = []

        if use_pose:
            raw_pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
                if results.pose_landmarks else np.zeros(33 * 3)
            normalized_pose = normalize_spatial_points(raw_pose, anchor, scale)
            features_to_combine.append(normalized_pose)

        if use_face:
            raw_face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
                if results.face_landmarks else np.zeros(468 * 3)
            normalized_face = normalize_spatial_points(raw_face, anchor, scale)
            features_to_combine.append(normalized_face)

        if use_hands:
            raw_left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3)
            raw_right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
                if results.right_hand_landmarks else np.zeros(21 * 3)
            
            normalized_left = normalize_spatial_points(raw_left_hand, anchor, scale)
            normalized_right = normalize_spatial_points(raw_right_hand, anchor, scale)
            
            features_to_combine.append(normalized_left)
            features_to_combine.append(normalized_right)

        if features_to_combine:
            frame_vector = np.concatenate(features_to_combine)
            sequence_history.append(frame_vector)

    capture.release()

    if not sequence_history:
        # In case no landmarks were detected in any frame, return a zero tensor
        return np.zeros((target_frames, FRAME_FEATURES_DIM))

    # Temporal normalization: uniform subsampling to ensure a fixed number of frames
    return uniform_subsampling(sequence_history, target_frames=target_frames)


def run_extraction_pipeline(
        source_dir: str, 
        dest_dir: str,
        use_pose: bool,
        use_hands: bool,
        use_face: bool,
        target_frames: int
    ) -> None:
    """
    Orchestrates the massive dataset conversion by iterating through class folders,
    processing video files, and storing clean NumPy arrays.

    Args:
        source_dir (str): Root path of the new MP4 video dataset.
        dest_dir (str): Path where normalized .npy files will be saved.
        hand_landmarks (bool): Whether to extract hand landmarks.
        pose_landmarks (bool): Whether to extract pose landmarks.
    """
    os.makedirs(dest_dir, exist_ok=True)
    sign_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    mp_holistic = mp.solutions.holistic

    # Initialize the MediaPipe context optimized for batch processing
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

                # Avoid reprocessing if the script is interrupted
                if os.path.exists(final_save_path):
                    continue

                try:
                    landmarks_tensor = process_video_to_landmarks(
                        video_path=video_path, 
                        holistic_model=holistic, 
                        target_frames=target_frames,
                        use_pose=use_pose,
                        use_hands=use_hands,
                        use_face=use_face
                    )
                    if landmarks_tensor.shape[0] == target_frames:
                        np.save(final_save_path, landmarks_tensor)
                except Exception as process_error:
                    print(f"\n[!] Critical error in file {npy_filename}: {process_error}")


if __name__ == "__main__":
    print(f"Starting extraction:")
    print(f" > Extract Pose: {USE_POSE}")
    print(f" > Extract Hands: {USE_HANDS}")
    print(f" > Expected features per frame: {FRAME_FEATURES_DIM} features.")
    print(f" > Output temporal sequence: {TARGET_FRAMES} frames.")
    
    run_extraction_pipeline(
        source_dir=DATASET_VIDEOS_DIR, 
        dest_dir=DATASET_NPY_DIR,
        use_pose=USE_POSE,
        use_hands=USE_HANDS,
        use_face=USE_FACE,
        target_frames=TARGET_FRAMES
    )