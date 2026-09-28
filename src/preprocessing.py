import os
import glob
import cv2
import numpy as np
from typing import List
from tqdm import tqdm

from holistic_web import HolisticWebSession, HolisticWebError, is_session_dead
from landmarks import extract_frame_vector
from utils import (
    interpolate_zero_frames,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)
from config import (
    DATASET_VIDEOS_DIR,
    DATASET_NPY_DIR,
    MAX_FRAMES,
    USE_FACE,
    USE_POSE,
    USE_HANDS,
    FRAME_FEATURES_DIM,
    POSE_DIM,
    MEDIAPIPE_WEB_ID,
)


def process_video_to_landmarks(
    video_path: str,
    session: HolisticWebSession,
    target_frames: int,
    use_pose: bool,
    use_hands: bool,
    use_face: bool,
    left_handed: bool = False,
) -> np.ndarray:
    """
    Extrae landmarks frame a frame con Holistic JS/WASM (GPU), interpola ceros,
    recorta gesto (trim) y subsamplea — mismo pipeline lógico que camera.py.
    """
    capture = cv2.VideoCapture(video_path)
    sequence_history: List[np.ndarray] = []
    session.reset()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = session.process_bgr(frame)
        frame_vector = extract_frame_vector(results, use_pose, use_hands, use_face)
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


def _npy_is_usable(path: str, target_frames: int) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return arr.shape == (target_frames, FRAME_FEATURES_DIM) and bool(np.any(arr))


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

    with HolisticWebSession(require_gpu=True) as session:
        failed = 0
        saved = 0
        for sign_class in sign_classes:
            print(f"\n[*] Processing category: {sign_class}")
            class_input_path = os.path.join(source_dir, sign_class)
            class_output_path = os.path.join(dest_dir, sign_class)
            os.makedirs(class_output_path, exist_ok=True)

            available_videos = glob.glob(os.path.join(class_input_path, "*.mp4"))

            for video_path in tqdm(available_videos, desc="Video Progress"):
                npy_filename = os.path.basename(video_path).replace(".mp4", ".npy")
                final_save_path = os.path.join(class_output_path, npy_filename)

                if _npy_is_usable(final_save_path, target_frames) and not force_reprocess:
                    continue

                last_error = None
                for attempt in range(3):
                    try:
                        landmarks_tensor = process_video_to_landmarks(
                            video_path=video_path,
                            session=session,
                            target_frames=target_frames,
                            use_pose=use_pose,
                            use_hands=use_hands,
                            use_face=use_face,
                            left_handed=_is_left_handed_video(video_path),
                        )
                        if landmarks_tensor.shape == (target_frames, FRAME_FEATURES_DIM) and np.any(
                            landmarks_tensor
                        ):
                            np.save(final_save_path, landmarks_tensor)
                            saved += 1
                            session.maybe_recycle(every_videos=25)
                            last_error = None
                            break
                        last_error = RuntimeError("secuencia vacía (Holistic no detectó pose/manos)")
                        break
                    except Exception as process_error:
                        last_error = process_error
                        if is_session_dead(process_error) and attempt < 2:
                            print(
                                f"\n[!] Chrome/Holistic se cayó en {npy_filename}. "
                                f"Reinicio y reintento ({attempt + 1}/2)."
                            )
                            try:
                                session.recycle("crash")
                            except HolisticWebError as recycle_error:
                                print(f"[!] No se pudo reiniciar: {recycle_error}")
                                break
                            continue
                        break
                if last_error is not None:
                    failed += 1
                    print(f"\n[!] Critical error in file {npy_filename}: {last_error}")

        print(f"\n[*] Listo. Guardados: {saved}. Fallidos: {failed}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrae landmarks LSA desde videos MP4 con MediaPipe Holistic JS/WASM (GPU)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa aunque el .npy ya exista (necesario tras cambiar trim/interpolación).",
    )
    args = parser.parse_args()

    print("Starting extraction:")
    print(f" > MediaPipe: {MEDIAPIPE_WEB_ID} (WebGL GPU, no Python)")
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
