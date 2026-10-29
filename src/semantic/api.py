"""HTTP: GET /health, POST /sign, POST /hearing."""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from semantic.config import AUTH_TOKEN
from semantic.service import (
    health_snapshot,
    infer_slot,
    load_models,
    translate_hearing,
    translate_sign,
)

app = FastAPI(title="LSA semantic", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignIn(BaseModel):
    glosses: str = Field(..., min_length=1, max_length=2000)


class HearingIn(BaseModel):
    spanish: str = Field(..., min_length=1, max_length=2000)


def _check_token(authorization: Optional[str], x_lsa_token: Optional[str]) -> None:
    expected = AUTH_TOKEN or os.getenv("LSA_SEMANTIC_TOKEN", "").strip()
    if not expected:
        return
    got = ""
    if authorization:
        got = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else authorization.strip()
        )
    if not got and x_lsa_token:
        got = x_lsa_token.strip()
    if got != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
def health():
    return health_snapshot()


def _timed_infer(channel: str, run):
    """Queue wait + GGUF inference. `ms` stays as inference (RNF-01)."""
    with infer_slot() as queued_s:
        t_inicio = time.perf_counter()
        try:
            text, model_used = run()
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        inference_ms = (time.perf_counter() - t_inicio) * 1000
    queue_ms = queued_s * 1000
    print(
        f"[semantic] Inference (SLM) [RNF-01]: t_inferencia={inference_ms:.1f} ms "
        f"channel={channel} (t_inicio_inferencia → t_fin_inferencia)",
        flush=True,
    )
    return text, model_used, round(queue_ms, 1), round(inference_ms, 1)


def _run_sign(body: SignIn, authorization, x_lsa_token):
    _check_token(authorization, x_lsa_token)
    spanish, model_used, queue_ms, inference_ms = _timed_infer(
        "sign", lambda: translate_sign(body.glosses)
    )
    return {
        "spanish": spanish or "",
        "glosses": body.glosses.strip(),
        "model_used": model_used,
        "ms": inference_ms,
        "inference_ms": inference_ms,
        "queued_ms": queue_ms,
        "queue_ms": queue_ms,
    }


def _run_hearing(body: HearingIn, authorization, x_lsa_token):
    _check_token(authorization, x_lsa_token)
    glosses, model_used, queue_ms, inference_ms = _timed_infer(
        "hearing", lambda: translate_hearing(body.spanish)
    )
    return {
        "glosses": glosses or "",
        "spanish": body.spanish.strip(),
        "model_used": model_used,
        "ms": inference_ms,
        "inference_ms": inference_ms,
        "queued_ms": queue_ms,
        "queue_ms": queue_ms,
    }


@app.post("/sign")
@app.post("/translate")
def sign(
    body: SignIn,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
):
    return _run_sign(body, authorization, x_lsa_token)


@app.post("/hearing")
def hearing(
    body: HearingIn,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
):
    return _run_hearing(body, authorization, x_lsa_token)


def warmup() -> None:
    print("[semantic] Loading GGUF from src/semantic/models/…", flush=True)
    load_models()
    snap = health_snapshot()
    print(f"[semantic] sign RAM: {snap.get('sign_gguf')}", flush=True)
    print(f"[semantic] hearing RAM: {snap.get('hearing_gguf')}", flush=True)
    if not snap.get("ready"):
        raise RuntimeError("[semantic] Warmup incomplete: GGUF not in RAM")
    print("[semantic] Ready GET /health POST /sign POST /hearing", flush=True)
