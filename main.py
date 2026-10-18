import mimetypes
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

mimetypes.add_type("application/json", ".json")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("application/octet-stream", ".fbx")
mimetypes.add_type("application/octet-stream", ".pose")

BASE_DIR = Path(__file__).resolve().parent
POSES_DIR = BASE_DIR / "poses"

app = FastAPI(title="Avatar T-Pose Assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


if not POSES_DIR.is_dir():
    raise RuntimeError(f"No se encontró la carpeta de T-Pose: {POSES_DIR}")

app.mount("/tposes", StaticFiles(directory=str(POSES_DIR)), name="tposes")

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
