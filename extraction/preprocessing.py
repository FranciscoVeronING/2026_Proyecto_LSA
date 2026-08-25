import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Any, Optional
from tqdm import tqdm

from extraction.config import (
    DATASET_NPY_DIR,
    DATASET_VIDEOS_DIR,
    FRAME_FEATURES_DIM,
    MAX_FRAMES,
    POSE_DIM,
    USE_FACE,
    USE_HANDS,
    USE_POSE,
)
from extraction.utils import (
    get_anchor_and_scale,
    normalize_spatial_points,
    interpolate_zero_frames,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)


def _extract_frame_vector(results, use_pose: bool, use_hands: bool, use_face: bool) -> Optional[np.ndarray]:
    anchor, scale = get_anchor_and_scale(results.pose_landmarks)
    features_to_combine = []

    if use_pose:
        raw_pose = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 3)
        )
        features_to_combine.append(normalize_spatial_points(raw_pose, anchor, scale))

    if use_face:
        raw_face = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks
            else np.zeros(468 * 3)
        )
        features_to_combine.append(normalize_spatial_points(raw_face, anchor, scale))

    if use_hands:
        raw_left_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        raw_right_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )
        features_to_combine.append(normalize_spatial_points(raw_left_hand, anchor, scale))
        features_to_combine.append(normalize_spatial_points(raw_right_hand, anchor, scale))

    if not features_to_combine:
        return None
    return np.concatenate(features_to_combine)


def process_video_to_landmarks(
    video_path: str,
    holistic_model: Any,
    target_frames: int,
    use_pose: bool,
    use_hands: bool,
    use_face: bool,
    left_handed: bool = False,
) -> np.ndarray:
    """
    Extrae landmarks frame a frame, interpola ceros, recorta gesto (trim)
    y subsamplea — mismo pipeline lógico que camera.py.
    """
    capture = cv2.VideoCapture(video_path)
    sequence_history: List[np.ndarray] = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(rgb_image)
        frame_vector = _extract_frame_vector(results, use_pose, use_hands, use_face)
        if frame_vector is not None:
            sequence_history.append(frame_vector)

    capture.release()

    if not sequence_history:
        return np.zeros((target_frames, FRAME_FEATURES_DIM), dtype=np.float32)

    sequence_history = interpolate_zero_frames(sequence_history)

    if left_handed:
        sequence_history = [
            mirror_landmarks_for_left_handed(frame, pose_dim=POSE_DIM)
            for frame in sequence_history
        ]

    return sequence_buffer_to_model_input(sequence_history, target_frames=target_frames)


def _is_left_handed_video(video_path: str) -> bool:
    """Convención opcional: sufijo _zurdo o _left en el nombre del archivo."""
    name = os.path.basename(video_path).lower()
    return "_zurdo" in name or "_left" in name


def run_extraction_pipeline(
    source_dir: str,
    dest_dir: str,
    use_pose: bool,
    use_hands: bool,
    use_face: bool,
    target_frames: int,
    force_reprocess: bool = False,
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
                npy_filename = os.path.basename(video_path).replace(".mp4", ".npy")
                final_save_path = os.path.join(class_output_path, npy_filename)

                if os.path.exists(final_save_path) and not force_reprocess:
                    continue

                try:
                    landmarks_tensor = process_video_to_landmarks(
                        video_path=video_path,
                        holistic_model=holistic,
                        target_frames=target_frames,
                        use_pose=use_pose,
                        use_hands=use_hands,
                        use_face=use_face,
                        left_handed=_is_left_handed_video(video_path),
                    )
                    if landmarks_tensor.shape == (target_frames, FRAME_FEATURES_DIM):
                        np.save(final_save_path, landmarks_tensor)
                except Exception as process_error:
                    print(f"\n[!] Critical error in file {npy_filename}: {process_error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrae landmarks LSA desde videos MP4.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa aunque el .npy ya exista (necesario tras cambiar trim/interpolación).",
    )
    args = parser.parse_args()

    print("Starting extraction:")
    print(f" > Extract Pose: {USE_POSE}")
    print(f" > Extract Hands: {USE_HANDS}")
    print(f" > Pipeline: interpolate → trim → subsample (aligned with camera.py)")
    print(f" > Expected features per frame: {FRAME_FEATURES_DIM}")
    print(f" > Output temporal sequence: {MAX_FRAMES} frames")
    if args.force:
        print(" > Modo: FORCE (reprocesar todos los videos)")

    run_extraction_pipeline(
        source_dir=DATASET_VIDEOS_DIR,
        dest_dir=DATASET_NPY_DIR,
        use_pose=USE_POSE,
        use_hands=USE_HANDS,
        use_face=USE_FACE,
        target_frames=MAX_FRAMES,
        force_reprocess=args.force,
    )
