"""FastAPI del traductor: POST /sign y POST /hearing."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

from semantic.config import (
    BACKUP_GGUF,
    OYENTE_GGUF_DIR,
    PRIMARY_GGUF,
    SEMANTIC_TOKEN,
    SORDO_GGUF_DIR,
)
from semantic.models import describe_slot
from semantic.translator import (
    infer_slot,
    load_model_and_tokenizer,
    load_status,
    loaded_paths,
    translate_glosses,
    translate_spanish_to_glosses,
)

app = FastAPI(title="LSA semántico", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SordoIn(BaseModel):
    glosses: str = Field(..., min_length=1, max_length=2000)


class OyenteIn(BaseModel):
    spanish: str = Field(..., min_length=1, max_length=2000)


def _check_token(authorization: Optional[str], x_lsa_token: Optional[str]) -> None:
    expected = SEMANTIC_TOKEN or os.getenv("LSA_SEMANTIC_TOKEN", "").strip()
    if not expected:
        return
    got = ""
    if authorization:
        got = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
    if not got and x_lsa_token:
        got = x_lsa_token.strip()
    if got != expected:
        raise HTTPException(status_code=401, detail="Token inválido")


@app.get("/health")
def health():
    """ILSA: listo solo con HTTP 200 y ready=true. Si el GGUF aún no está en RAM,
    el puerto no está abierto (cold start) o este campo es false."""
    paths = loaded_paths()
    sordo_slot = describe_slot(SORDO_GGUF_DIR)
    oyente_slot = describe_slot(OYENTE_GGUF_DIR)
    ready = bool(paths.get("sordo"))
    status = load_status()
    return {
        "ok": True,
        "ready": ready,
        "loading": status["loading"],
        "load_error": status["error"],
        "primary": PRIMARY_GGUF,
        "backup": BACKUP_GGUF,
        "sordo": sordo_slot,
        "oyente": oyente_slot,
        "sordo_gguf": paths.get("sordo"),
        "oyente_gguf": paths.get("oyente"),
        "model_used_sordo": paths.get("sordo_model"),
        "model_used_oyente": paths.get("oyente_model"),
    }


def _sordo_impl(body: SordoIn, authorization, x_lsa_token):
    _check_token(authorization, x_lsa_token)
    with infer_slot() as queued_s:
        t0 = time.perf_counter()
        try:
            spanish, model_used = translate_glosses(body.glosses)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        spanish = spanish or ""
        ms = (time.perf_counter() - t0) * 1000
    return {
        "spanish": spanish,
        "glosses": body.glosses.strip(),
        "model_used": model_used,
        "ms": round(ms, 1),
        "queued_ms": round(queued_s * 1000, 1),
    }


def _oyente_impl(body: OyenteIn, authorization, x_lsa_token):
    _check_token(authorization, x_lsa_token)
    with infer_slot() as queued_s:
        t0 = time.perf_counter()
        try:
            glosses, model_used = translate_spanish_to_glosses(body.spanish)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        glosses = glosses or ""
        ms = (time.perf_counter() - t0) * 1000
    return {
        "glosses": glosses,
        "spanish": body.spanish.strip(),
        "model_used": model_used,
        "ms": round(ms, 1),
        "queued_ms": round(queued_s * 1000, 1),
    }


@app.post("/sign")
def sign(
    body: SordoIn,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
):
    return _sordo_impl(body, authorization, x_lsa_token)


@app.post("/hearing")
def hearing(
    body: OyenteIn,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
):
    return _oyente_impl(body, authorization, x_lsa_token)


def warmup() -> None:
    print("[semantic-http] Cargando GGUF desde src/semantic/models/…", flush=True)
    load_model_and_tokenizer()
    paths = loaded_paths()
    print(f"[semantic-http] sordo RAM: {paths.get('sordo')}", flush=True)
    print(f"[semantic-http] oyente RAM: {paths.get('oyente')}", flush=True)
    if not paths.get("sordo"):
        raise RuntimeError("[semantic-http] Warmup incompleto: sin GGUF en RAM")
    print("[semantic-http] Listo /sign /hearing", flush=True)
