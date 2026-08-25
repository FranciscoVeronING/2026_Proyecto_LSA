import tempfile
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import mediapipe as mp

# Importamos directamente todo desde tu preprocessing.py
from preprocessing import (
    FRAME_FEATURES_DIM,
    MAX_FRAMES,
    USE_FACE,
    USE_HANDS,
    USE_POSE,
    process_video_to_landmarks,
)
from utils import CROP_H, CROP_W, CROP_X, CROP_Y

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def decode_qr_image(image_path: Path) -> str:
    """Lee la imagen usando np.fromfile para evitar errores con 'ñ' en Windows."""
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"No se pudo abrir la imagen QR: {image_path.name}")

    detector = cv2.QRCodeDetector()
    value, _, _ = detector.detectAndDecode(image)
    if not value:
        raise ValueError(f"No se encontró un QR legible en: {image_path.name}")
    return value.strip()


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or hostname not in YOUTUBE_HOSTS:
        raise ValueError(f"El QR no contiene una URL de YouTube válida: {url}")
    return url


def download_youtube_video(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError(
            "Falta yt-dlp. Instalá con: py -m pip install yt-dlp"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    # Descarga directa del video MP4 (sin requerir fusión de audio ni FFmpeg externo)
    options = {
        "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "youtube_video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": "cookies.txt",
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])

    candidates = sorted(output_dir.glob("youtube_video.*"))
    if not candidates:
        raise RuntimeError("yt-dlp no generó ningún archivo de video")
    return candidates[0]


def crop_video(input_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    if CROP_X + CROP_W > width or CROP_Y + CROP_H > height:
        capture.release()
        raise ValueError(
            f"El recorte ({CROP_W}x{CROP_H} desde {CROP_X},{CROP_Y}) "
            f"excede la resolución {width}x{height}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CROP_W, CROP_H)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("No se pudo inicializar VideoWriter para el recorte")

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            cropped = frame[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W]
            writer.write(cropped)
    finally:
        capture.release()
        writer.release()


def extract_landmarks(video_path: Path, output_path: Path, holistic_model) -> None:
    """Ejecuta la función del preprocessing original pasando el modelo ya instanciado."""
    landmarks = process_video_to_landmarks(
        video_path=str(video_path),
        holistic_model=holistic_model,
        target_frames=MAX_FRAMES,
        use_pose=USE_POSE,
        use_hands=USE_HANDS,
        use_face=USE_FACE,
        left_handed=False,
    )

    expected_shape = (MAX_FRAMES, FRAME_FEATURES_DIM)
    if landmarks.shape != expected_shape:
        raise RuntimeError(
            f"Forma inesperada de landmarks: {landmarks.shape}; se esperaba {expected_shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, landmarks)


def run_pipeline(
    qr_path: Path, scenario: int, output_path: Path, holistic_model, keep_video: bool = False
) -> Path:
    url = validate_youtube_url(decode_qr_image(qr_path))
    
    with tempfile.TemporaryDirectory(prefix="qr_video_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        downloaded_path = download_youtube_video(url, temporary_path)
        source_path = downloaded_path

        if scenario == 1:
            source_path = temporary_path / "cropped.mp4"
            crop_video(downloaded_path, source_path)

        extract_landmarks(source_path, output_path, holistic_model)

        if keep_video:
            saved_video = output_path.with_suffix(".mp4")
            saved_video.write_bytes(source_path.read_bytes())

    return output_path


def procesar_carpeta(
    carpeta_input: Path, carpeta_output: Path, escenario: int, holistic_model, keep_video: bool = False
) -> None:
    if not carpeta_input.exists():
        print(f"⚠ Carpeta no encontrada: {carpeta_input.resolve()}")
        return

    carpeta_output.mkdir(parents=True, exist_ok=True)
    imagenes = sorted(carpeta_input.glob("*.png"))

    if not imagenes:
        print(f"ℹ No se encontraron imágenes .png en {carpeta_input}")
        return

    print(f"\n{'='*60}")
    print(f"Iniciando: {carpeta_input.name} -> {carpeta_output.name} (Escenario {escenario})")
    print(f"Total a procesar: {len(imagenes)} imágenes")
    print(f"{'='*60}")

    for idx, img_path in enumerate(imagenes, start=1):
        salida_npy = carpeta_output / f"avatar_{img_path.stem}.npy"

        if salida_npy.exists():
            #print(f"[{idx}/{len(imagenes)}] Ya existe: {salida_npy.name} (Omitiendo)")
            continue

        print(f"[{idx}/{len(imagenes)}] Procesando {img_path.name}...")
        try:
            run_pipeline(
                qr_path=img_path,
                scenario=escenario,
                output_path=salida_npy,
                holistic_model=holistic_model,
                keep_video=keep_video,
            )
            #print(f"  ✔ Guardado en: {salida_npy.name}")
        except Exception as error:
            print(f"  ✖ Error procesando {img_path.name}: {error}")


def main() -> None:
    # Importación resiliente de Holistic
    try:
        mp_holistic = mp.solutions.holistic
    except AttributeError:
        import mediapipe.python.solutions.holistic as mp_holistic

    # Instanciamos Holistic una sola vez para todo el procesamiento por lotes (mucho más rápido)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # 1. Procesa señario1 (con recorte)
        procesar_carpeta(
            carpeta_input=Path("señario1"),
            carpeta_output=Path("señario1_landmarks"),
            escenario=1,
            holistic_model=holistic,
            keep_video=False,
        )

        # 2. Procesa Señario2 (video completo)
        procesar_carpeta(
            carpeta_input=Path("Señario2"),
            carpeta_output=Path("Señario2_landmarks"),
            escenario=2,
            holistic_model=holistic,
            keep_video=False,
        )

    print("\n🎉 ¡Procesamiento por lotes finalizado!")


if __name__ == "__main__":
    main()