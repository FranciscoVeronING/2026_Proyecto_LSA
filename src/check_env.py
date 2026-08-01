"""
Verifica que el entorno Python tenga versiones compatibles con el pipeline LSA.

Uso:
    cd src
    python check_env.py
"""
from __future__ import annotations

import sys


def _parse_version(version_str: str) -> tuple[int, ...]:
    parts = []
    for piece in version_str.split(".")[:3]:
        try:
            parts.append(int("".join(c for c in piece if c.isdigit()) or "0"))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _version_in_range(version_str: str, low: str, high: str) -> bool:
    v = _parse_version(version_str)
    lo = _parse_version(low)
    hi = _parse_version(high)
    return lo <= v < hi


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    py = sys.version_info
    print(f"Python: {py.major}.{py.minor}.{py.micro}")

    if py < (3, 10):
        warnings.append(
            "Python < 3.10 detectado. Recomendado: 3.11.x "
            "(el codigo usa typing compatible, pero 3.11 es mas estable con las deps)."
        )
    if py >= (3, 13):
        warnings.append(
            "Python 3.13+ puede no tener wheels oficiales para mediapipe 0.10.21. "
            "Use Python 3.11 si hay problemas."
        )

    # protobuf (debe instalarse antes de mediapipe en el check)
    try:
        import google.protobuf as pb

        pb_ver = pb.__version__
        print(f"protobuf: {pb_ver}")
        if not _version_in_range(pb_ver, "4.25.3", "5.0.0"):
            errors.append(
                f"protobuf {pb_ver} incompatible. MediaPipe 0.10.21 requiere "
                f"protobuf>=4.25.3,<5.\n"
                f"  Fix: pip install 'protobuf>=4.25.3,<5'"
            )
    except ImportError:
        errors.append("protobuf no instalado. pip install 'protobuf>=4.25.3,<5'")

    # numpy
    try:
        import numpy as np

        np_ver = np.__version__
        print(f"numpy: {np_ver}")
        major = int(np_ver.split(".")[0])
        if major >= 2:
            errors.append(
                f"numpy {np_ver} incompatible. mediapipe 0.10.21 requiere numpy<2.\n"
                f"  Fix: pip install 'numpy>=1.26,<2'"
            )
    except ImportError:
        errors.append("numpy no instalado.")

    # mediapipe + solutions API
    try:
        import mediapipe as mp

        mp_ver = getattr(mp, "__version__", "unknown")
        print(f"mediapipe: {mp_ver}")

        mp_tuple = _parse_version(mp_ver)
        if mp_tuple >= (0, 10, 30):
            errors.append(
                f"mediapipe {mp_ver} elimino mp.solutions (Holistic).\n"
                f"  Fix: pip install mediapipe==0.10.21"
            )
        elif mp_tuple > (0, 10, 21):
            warnings.append(
                f"mediapipe {mp_ver}: no probado. Recomendado pin: mediapipe==0.10.21"
            )

        if not hasattr(mp, "solutions"):
            errors.append(
                "mediapipe no expone 'solutions'. El proyecto usa mp.solutions.holistic.\n"
                "  Fix: pip install mediapipe==0.10.21"
            )
        else:
            holistic = mp.solutions.holistic
            with holistic.Holistic(min_detection_confidence=0.5) as model:
                pass
            print("mediapipe.solutions.holistic: OK")
    except ImportError:
        errors.append("mediapipe no instalado. pip install mediapipe==0.10.21")
    except Exception as exc:
        errors.append(f"mediapipe instalado pero falla al iniciar Holistic: {exc}")

    # torch (opcional para preprocessing, requerido para train/camera)
    try:
        import torch

        print(f"torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        warnings.append("torch no instalado (necesario para train.py y camera.py).")

    try:
        import cv2

        print(f"opencv: {cv2.__version__}")
    except ImportError:
        errors.append("opencv-python no instalado.")

    print()
    for msg in warnings:
        print(f"[WARN] {msg}")
    for msg in errors:
        print(f"[ERROR] {msg}")

    if errors:
        print()
        print("Entorno NO listo. Stack recomendado:")
        print("  Python 3.11 + mediapipe==0.10.21 + protobuf>=4.25.3,<5 + numpy<2")
        print("Ver environment.yml o requirements.txt en la raiz del repo.")
        return 1

    print()
    print("Entorno OK para preprocessing / train / camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
