import mimetypes
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
POSES_DIR = ROOT_DIR / "poses"
PORT = int(os.environ.get("PORT", 8080))
MAX_LETTERS_PER_REQUEST = 20
FINGERSPELL_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"ENIE", "CH"}

mimetypes.add_type("application/json", ".json")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("application/octet-stream", ".fbx")
mimetypes.add_type("application/octet-stream", ".pose")
