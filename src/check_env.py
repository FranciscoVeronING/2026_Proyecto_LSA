"""
Verifica que el entorno tenga Python + Node (Holistic JS/WASM GPU).

Uso:
    cd src
    python check_env.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    py = sys.version_info
    print(f"Python: {py.major}.{py.minor}.{py.micro}")

    if py < (3, 10):
        warnings.append(
            "Python < 3.10 detectado. Recomendado: 3.11.x."
        )

    try:
        import numpy as np

        print(f"numpy: {np.__version__}")
    except ImportError:
        errors.append("numpy no instalado.")

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

    try:
        import mediapipe as mp

        warnings.append(
            f"mediapipe Python {getattr(mp, '__version__', '?')} está instalado "
            "pero ya no se usa. El clasificador extrae landmarks con "
            "@mediapipe/holistic JS/WASM. Podés desinstalarlo: pip uninstall mediapipe"
        )
    except ImportError:
        print("mediapipe Python: no instalado (correcto)")

    node = shutil.which("node")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not node:
        errors.append("Node.js no está en PATH. Instalá Node 18+.")
    else:
        try:
            node_ver = subprocess.check_output([node, "-v"], text=True).strip()
            print(f"node: {node_ver}")
            major = int("".join(c for c in node_ver.split(".")[0] if c.isdigit()) or "0")
            if major < 18:
                errors.append(f"Node {node_ver} es viejo. Hace falta Node 18+.")
        except subprocess.CalledProcessError:
            errors.append("node -v falló.")
    if not npm:
        errors.append("npm no está en PATH.")
    else:
        print(f"npm: {npm}")

    if node and npm and not errors:
        try:
            from holistic_web import HolisticWebSession, ensure_installed

            ensure_installed()
            print("holistic_web npm: OK")
            print("Probando Holistic JS/WASM con GPU (Chrome/WebGL)...")
            with HolisticWebSession(require_gpu=True) as session:
                gpu = session.gpu_info
                print(
                    f"mediapipe web: @mediapipe/holistic@0.5.1675471629 | "
                    f"{gpu.get('vendor', '?')} / {gpu.get('renderer', '?')}"
                )
        except Exception as exc:
            errors.append(f"Holistic Web GPU no arrancó: {exc}")

    print()
    for msg in warnings:
        print(f"[WARN] {msg}")
    for msg in errors:
        print(f"[ERROR] {msg}")

    if errors:
        print()
        print("Entorno NO listo. Stack recomendado:")
        print("  Python 3.11 + Node 18+ + Chrome/Edge con WebGL GPU")
        print("  cd src/holistic_web && npm install && npx playwright install chromium")
        print("Ver README.md")
        return 1

    print()
    print("Entorno OK para preprocessing / train / camera (MediaPipe Web GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
