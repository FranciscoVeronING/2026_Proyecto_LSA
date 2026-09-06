"""Descarga MediaPipe Holistic (WASM) para la extensión y genera íconos PNG."""

from __future__ import annotations

import struct
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "extension" / "vendor" / "mediapipe"
ICONS = REPO / "extension" / "icons"
BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/"
FILES = [
    "holistic.js",
    "holistic.binarypb",
    "holistic_solution_packed_assets.data",
    "holistic_solution_packed_assets_loader.js",
    "holistic_solution_simd_wasm_bin.js",
    "holistic_solution_simd_wasm_bin.wasm",
    "holistic_solution_wasm_bin.js",
    "holistic_solution_wasm_bin.wasm",
    "pose_landmark_lite.tflite",
]


def write_png(path: Path, size: int, rgb: tuple[int, int, int]) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    ICONS.mkdir(parents=True, exist_ok=True)
    write_png(ICONS / "icon16.png", 16, (255, 159, 28))
    write_png(ICONS / "icon48.png", 48, (255, 159, 28))
    write_png(ICONS / "icon128.png", 128, (255, 159, 28))
    print("Iconos OK")
    for name in FILES:
        dest = VENDOR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Ya existe {name}")
            continue
        url = BASE + name
        print(f"Descargando {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
