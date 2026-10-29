"""Local API for the extension.

Meet fetches with origin meet.google.com (content script), not chrome-extension://.
That is why CORS includes Meet. Do not bind 0.0.0.0: the model would sit on the LAN.
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

from backend.http_schemas import ClientMetricsIn, HearingIn, SessionIn, SignIn, SttIn, TtsIn, UtteranceEndIn

if getattr(sys, "frozen", False):
    REPO_ROOT = Path(sys.executable).resolve().parent
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
_EXE_NAMES = ("ILSA.exe", "LSABackend.exe")
_EXE_DIRS = (
    REPO_ROOT / "dist" / "LSABackend",
    REPO_ROOT / "dist",
    REPO_ROOT / "packaging" / "dist" / "LSABackend",
)
EXE_CANDIDATES = [d / n for d in _EXE_DIRS for n in _EXE_NAMES]

app = FastAPI(title="ILSA", version="1.0.0")
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


def _health_metrics() -> str:
    try:
        from backend.metrics import health_line

        return health_line()
    except Exception:
        return ""


def get_session():
    global _session
    if _session is None:
        raise HTTPException(status_code=503, detail="El backend todavía no inicializó el pipeline.")
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
        if session and getattr(session, "semantic_error", ""):
            semantic_label = "Translator unavailable"
            semantic_load = ""
        elif session and not getattr(session, "semantic_ready", False):
            semantic_label = "No translator"
            semantic_load = ""
        else:
            semantic_label = "Remote translator"
            semantic_load = "cloud"
    return {
        "ok": session is not None,
        "mode": _mode,
        "classifier_ready": bool(session and getattr(session, "model", None) is not None),
        "num_classes": len(getattr(session, "idx_to_class", {}) or {}) if session else 0,
        "semantic_ready": bool(session and getattr(session, "semantic_ready", False)),
        "semantic_error": getattr(session, "semantic_error", "") if session else "",
        "semantic_model": model_id,
        "semantic_label": semantic_label,
        "semantic_load": semantic_load,
        "device": str(getattr(session, "device", "")) if session else "",
        "left_handed": bool(session and getattr(session, "left_handed", False)),
        "metrics": _health_metrics(),
    }


@app.get("/config")
def capture_config():
    return get_session().capture_config()


@app.get("/state")
def state():
    return get_session().snapshot()


@app.get("/metrics")
def metrics_summary():
    from backend.metrics import summary

    return summary()


@app.post("/session")
def open_session(body: SessionIn):
    session = get_session()
    session.reset_session(left_handed=getattr(session, "left_handed", False))
    return session.snapshot()


@app.post("/sign")
def ingest_sign(body: SignIn):
    # Landmarks, not JPEG. Holistic runs in the extension.
    if _mode == "hearing":
        raise HTTPException(status_code=400, detail="Modo oyente: no se clasifican señas.")
    if not body.frames:
        print("[backend] POST /sign: empty body")
        raise HTTPException(status_code=400, detail="Faltan frames de la seña.")
    print(f"[backend] POST /sign: {len(body.frames)} frames")
    return get_session().ingest_sign(body.frames, client=body.client)


@app.post("/stt")
def stt_wav(body: SttIn):
    if _mode != "hearing":
        raise HTTPException(status_code=400, detail="Solo modo oyente.")
    import base64

    raw = (body.wav_b64 or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Falta el audio.")
    try:
        wav = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="WAV inválido") from exc
    from backend.stt import transcribe_wav

    try:
        text = transcribe_wav(wav, lang=body.lang or "es-ES")
    except Exception as exc:
        print(f"[backend] /stt error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    print(f"[backend] /stt → {text!r}")
    return {"spanish": text, "ok": True}


@app.post("/tts")
def tts_spanish(body: TtsIn):
    """WAV de la traducción, para mezclarlo en el audio que Meet envía."""
    if _mode == "hearing":
        raise HTTPException(status_code=400, detail="La voz en off es del modo sordo.")
    import base64

    from backend.tts import synthesize_wav

    try:
        wav = synthesize_wav(body.text)
    except Exception as exc:
        print(f"[backend] /tts error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    print(f"[backend] /tts {len(wav)} bytes")
    return {"ok": True, "wav_b64": base64.b64encode(wav).decode("ascii")}


@app.post("/hearing")
def hearing_speech(body: HearingIn):
    if _mode != "hearing":
        raise HTTPException(status_code=400, detail="Solo modo oyente.")
    session = get_session()
    ingest = getattr(session, "ingest_speech", None)
    if ingest is None:
        raise HTTPException(status_code=400, detail="Esta sesión no acepta voz.")
    return ingest(body.spanish, final=body.final)


@app.post("/activity")
def activity():
    get_session().note_activity()
    return {"ok": True}


@app.post("/utterance/end")
def utterance_end(body: UtteranceEndIn | None = None):
    session = get_session()
    pause_ms = body.pause_ms if body is not None else None
    return session.close_utterance(pause_ms=pause_ms)


@app.post("/metrics/client")
def metrics_client(body: ClientMetricsIn):
    from backend.metrics import log_event
    from utils.benchmark_logger import add_delivery
    from utils.process_stats import snapshot as proc_snap

    if body.delivery_ms is not None:
        add_delivery(body.channel, body.delivery_ms)
    log_event(
        "client",
        channel=body.channel,
        delivery_ms=body.delivery_ms,
        pause_ms=body.pause_ms,
        fps=body.fps,
        holistic_fps=body.holistic_fps,
        rss_mb=body.rss_mb,
        **proc_snap(),
    )
    if body.fps is not None:
        print(f"[metrics] FPS captureStream={body.fps:.1f}", flush=True)
    if body.holistic_fps is not None:
        print(f"[metrics] FPS Holistic sandbox={body.holistic_fps:.1f}", flush=True)
    return {"ok": True, "process": proc_snap()}


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
        raise HTTPException(status_code=404, detail=f"Falta {safe}")
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
            filename="LSA-start-engine.bat",
            media_type="application/octet-stream",
        )
    raise HTTPException(
        status_code=404,
        detail="No hay .exe. En desarrollo usá: python run_backend.py",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LSA backend for the Chrome extension.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-semantic",
        "--no-llm",
        dest="no_semantic",
        action="store_true",
        help="Do not call the Cloud Run translator.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No ILSA window: uvicorn in the console only (dev / scripts).",
    )
    parser.add_argument(
        "--mode",
        choices=("signer", "hearing"),
        default="signer",
        help="signer = LSA→Spanish; hearing = voice→captions. Only applies with --headless.",
    )
    return parser.parse_args(argv)


def init_session(
    enable_semantic: bool = True,
    mode: str = "signer",
    semantic_model_id: str | None = None,
    left_handed: bool = False,
):
    global _session, _mode
    _mode = "hearing" if mode == "hearing" else "signer"
    if _mode == "hearing":
        from backend.hearing_session import HearingSession

        _session = HearingSession()
        return
    from backend.signer_session import SignerSession

    _session = SignerSession(
        enable_semantic=enable_semantic,
        semantic_model_id=semantic_model_id,
        left_handed=left_handed,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "[ilsa] Warning: the API should listen on localhost only. "
            f"host={args.host!r} exposes the classifier on the network."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("LSA_BACKEND", "1")
    if args.headless:
        init_session(enable_semantic=not args.no_semantic, mode=args.mode)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return
    from backend.window import launch

    launch(args)


if __name__ == "__main__":
    main()
