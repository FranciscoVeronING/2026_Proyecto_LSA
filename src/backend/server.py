"""API local para la extensión Chrome: señas → glosas → español."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if getattr(sys, "frozen", False):
    REPO_ROOT = Path(sys.executable).resolve().parent
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
EXE_CANDIDATES = [
    REPO_ROOT / "dist" / "LSABackend" / "IRIS.exe",
    REPO_ROOT / "dist" / "LSABackend" / "LSABackend.exe",
    REPO_ROOT / "dist" / "IRIS.exe",
    REPO_ROOT / "dist" / "LSABackend.exe",
    REPO_ROOT / "packaging" / "dist" / "LSABackend" / "IRIS.exe",
    REPO_ROOT / "packaging" / "dist" / "LSABackend" / "LSABackend.exe",
]

app = FastAPI(title="LSA Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    if request.url.path.startswith("/mediapipe"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

_session = None


class SessionIn(BaseModel):
    left_handed: bool = False


class SignIn(BaseModel):
    """Cuerpo de ``POST /sign``: landmarks por frame, no imagen."""

    frames: list[dict] = Field(default_factory=list)


def get_session():
    """Sesión global. 503 si ``main()`` todavía no llamó ``init_session``."""
    global _session
    if _session is None:
        raise HTTPException(status_code=503, detail="Backend aún no inicializó el pipeline.")
    return _session


@app.get("/health")
def health():
    """Liveness: pipeline creado, clasificador y LLM."""
    session = _session
    return {
        "ok": session is not None,
        "classifier_ready": bool(session and session.model is not None),
        "semantic_ready": bool(session and session.semantic_ready),
        "semantic_error": session.semantic_error if session else "",
        "device": str(session.device) if session else "",
    }


@app.get("/config")
def capture_config():
    """Umbrales de recorte para la extensión."""
    return get_session().capture_config()


@app.get("/state")
def state():
    """Snapshot sin mutar el buffer."""
    return get_session().snapshot()


@app.post("/session")
def open_session(body: SessionIn):
    """Reinicia glosas, español y mano dominante."""
    get_session().reset_session(left_handed=body.left_handed)
    return get_session().snapshot()


@app.post("/sign")
def ingest_sign(body: SignIn):
    """
    Una seña = lista de frames de **landmarks** (no JPEG) → glosa + snapshot.

    MediaPipe corre en la extensión. Este endpoint solo clasifica.
    """
    if not body.frames:
        print("[backend] POST /sign: body vacío")
        raise HTTPException(status_code=400, detail="Falta el conjunto de frames de la seña.")
    print(f"[backend] POST /sign: {len(body.frames)} frames")
    return get_session().ingest_sign(body.frames)


@app.post("/activity")
def activity():
    """Retrasa el cierre de enunciado mientras la persona sigue señando."""
    get_session().note_activity()
    return {"ok": True}


@app.post("/utterance/end")
def utterance_end():
    """Cierra la lista de glosas y traduce a español."""
    return get_session().close_utterance()


@app.post("/conversation/clear")
def conversation_clear():
    """Vacía el contexto conversacional de la LLM."""
    get_session().clear_conversation()
    return get_session().snapshot()


@app.get("/mediapipe/{name}")
def mediapipe_asset(name: str):
    """Modelos WASM/tflite para la extensión (evita fetch chrome-extension://)."""
    safe = Path(name).name
    path = (REPO_ROOT / "extension" / "vendor" / "mediapipe" / safe).resolve()
    root = (REPO_ROOT / "extension" / "vendor" / "mediapipe").resolve()
    if root not in path.parents and path.parent != root:
        raise HTTPException(status_code=400, detail="Nombre inválido")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No está {safe}")
    suffix = path.suffix.lower()
    media = {
        ".wasm": "application/wasm",
        ".js": "application/javascript",
        ".data": "application/octet-stream",
        ".tflite": "application/octet-stream",
        ".binarypb": "application/octet-stream",
    }.get(suffix, "application/octet-stream")
    return FileResponse(
        path,
        media_type=media,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "public, max-age=86400",
        },
    )


@app.get("/download/exe")
def download_exe():
    """Sirve LSABackend.exe o, en desarrollo, LSABackend.bat."""
    for path in EXE_CANDIDATES:
        if path.is_file():
            return FileResponse(
                path,
                filename="LSABackend.exe",
                media_type="application/octet-stream",
            )
    bat = REPO_ROOT / "LSABackend.bat"
    if bat.is_file():
        return FileResponse(
            bat,
            filename="LSA-iniciar-motor.bat",
            media_type="application/octet-stream",
        )
    raise HTTPException(
        status_code=404,
        detail="No hay .exe. En desarrollo usá: python run_backend.py",
    )


def parse_args(argv=None):
    """CLI de ``run_backend.py``: host, puerto, --no-llm, --gpu."""
    parser = argparse.ArgumentParser(description="Backend LSA para la extensión Chrome.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Intentar GPU para la LLM (Vulkan). Por defecto todo corre en CPU.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Sin ventana IRIS: solo uvicorn en consola (desarrollo / scripts).",
    )
    return parser.parse_args(argv)


def init_session(enable_llm: bool = True):
    """Construye el ``LSASession`` global (carga pesos; LLM en background)."""
    global _session
    from backend.session import LSASession

    _session = LSASession(enable_llm=enable_llm)


def main(argv=None):
    """Punto de entrada: ventana IRIS, o uvicorn solo con ``--headless``."""
    args = parse_args(argv)
    os.environ.setdefault("LSA_BACKEND", "1")
    if args.gpu:
        os.environ["LSA_USE_GPU"] = "1"
    if args.headless:
        init_session(enable_llm=not args.no_llm)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return
    from backend.iris_app import launch

    launch(args)


if __name__ == "__main__":
    main()
