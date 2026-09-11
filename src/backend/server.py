"""API local para la extensión.

Meet hace fetch con origin meet.google.com (content script), no chrome-extension://.
Por eso CORS incluye Meet. No bindear 0.0.0.0: queda el modelo en la LAN.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backend.http_schemas import SessionIn, SignIn

if getattr(sys, "frozen", False):
    REPO_ROOT = Path(sys.executable).resolve().parent
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
_EXE_NAMES = ("ILSA.exe", "IRIS.exe", "LSABackend.exe")
_EXE_DIRS = (
    REPO_ROOT / "dist" / "LSABackend",
    REPO_ROOT / "dist",
    REPO_ROOT / "packaging" / "dist" / "LSABackend",
)
EXE_CANDIDATES = [d / n for d in _EXE_DIRS for n in _EXE_NAMES]

app = FastAPI(title="LSA Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://meet.google.com",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ],
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
_mode = "signer"


def get_session():
    global _session
    if _session is None:
        raise HTTPException(status_code=503, detail="Backend aún no inicializó el pipeline.")
    return _session


@app.get("/health")
def health():
    session = _session
    model_id = ""
    if session and _mode == "signer":
        model_id = getattr(session, "_current_model_id", "") or ""
    semantic_label = ""
    semantic_load = ""
    if _mode == "signer":
        from semantic.models import compute_label, friendly_label

        if session and getattr(session, "semantic_error", ""):
            semantic_label = "Traductor no disponible"
            semantic_load = ""
        elif session and not getattr(session, "semantic_ready", False):
            semantic_label = "Cargando el traductor…"
            semantic_load = ""
        else:
            semantic_label = friendly_label(model_id or "llama-3.2-1b")
            semantic_load = compute_label(model_id)
    return {
        "ok": session is not None,
        "mode": _mode,
        "classifier_ready": bool(session and getattr(session, "model", None) is not None),
        "semantic_ready": bool(session and getattr(session, "semantic_ready", False)),
        "semantic_error": getattr(session, "semantic_error", "") if session else "",
        "semantic_model": model_id,
        "semantic_label": semantic_label,
        "semantic_load": semantic_load,
        "device": str(getattr(session, "device", "")) if session else "",
    }


@app.get("/semantic/models")
def semantic_models():
    from semantic.models import list_semantic_models

    session = _session
    active = ""
    if session and _mode == "signer":
        active = getattr(session, "_current_model_id", "") or ""
    return {"models": list_semantic_models(), "active": active}


@app.get("/config")
def capture_config():
    return get_session().capture_config()


@app.get("/state")
def state():
    return get_session().snapshot()


@app.post("/session")
def open_session(body: SessionIn):
    get_session().reset_session(left_handed=body.left_handed)
    return get_session().snapshot()


@app.post("/sign")
def ingest_sign(body: SignIn):
    # Landmarks, no JPEG. Holistic corre en la extensión.
    if _mode == "hearing":
        raise HTTPException(status_code=400, detail="Modo oyente: no se clasifican señas.")
    if not body.frames:
        print("[backend] POST /sign: body vacío")
        raise HTTPException(status_code=400, detail="Falta el conjunto de frames de la seña.")
    print(f"[backend] POST /sign: {len(body.frames)} frames")
    return get_session().ingest_sign(body.frames)


@app.post("/activity")
def activity():
    get_session().note_activity()
    return {"ok": True}


@app.post("/utterance/end")
def utterance_end():
    return get_session().close_utterance()


@app.post("/conversation/clear")
def conversation_clear():
    get_session().clear_conversation()
    return get_session().snapshot()


@app.get("/mediapipe/{name}")
def mediapipe_asset(name: str):
    safe = Path(name).name
    if safe != name or ".." in name or not safe:
        raise HTTPException(status_code=400, detail="Nombre inválido")
    allowed = {".wasm", ".js", ".data", ".tflite", ".binarypb"}
    if Path(safe).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
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
    for path in EXE_CANDIDATES:
        if path.is_file():
            return FileResponse(
                path,
                filename="ILSA.exe",
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
    parser = argparse.ArgumentParser(description="Backend LSA para la extensión Chrome.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Sin ventana ILSA: solo uvicorn en consola (desarrollo / scripts).",
    )
    parser.add_argument(
        "--mode",
        choices=("signer", "hearing"),
        default="signer",
        help="signer = LSA→español; hearing = voz→subtítulos. Solo aplica con --headless.",
    )
    return parser.parse_args(argv)


def init_session(enable_llm: bool = True, mode: str = "signer", semantic_model_id: str | None = None):
    global _session, _mode
    _mode = "hearing" if mode == "hearing" else "signer"
    if _mode == "hearing":
        from backend.hearing_session import HearingSession

        _session = HearingSession()
        return
    from backend.session import LSASession

    _session = LSASession(enable_llm=enable_llm, semantic_model_id=semantic_model_id)


def main(argv=None):
    args = parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "[ilsa] Advertencia: el API debería escuchar solo en localhost. "
            f"host={args.host!r} expone clasificador y LLM en la red."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("LSA_BACKEND", "1")
    if args.headless:
        if not args.no_llm and args.mode != "hearing":
            from semantic.gguf_fetch import ensure_gguf

            print("[ilsa] Comprobando traductor…")
            ensure_gguf(lambda _d, _t, msg: print(f"[ilsa] {msg}"))
        init_session(enable_llm=not args.no_llm, mode=args.mode)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return
    from backend.iris_app import launch

    launch(args)


if __name__ == "__main__":
    main()
