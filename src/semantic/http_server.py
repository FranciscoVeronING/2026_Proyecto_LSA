"""FastAPI del traductor (sordo y oyente). Corre en la PC del servidor, no en ILSA."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
import time

from semantic.config import DEFAULT_MODEL_ID, SEMANTIC_TOKEN, USE_CONVERSATION_HISTORY
from semantic.sessions import client_key, memory_for, n_clients
from semantic.translator import (
    get_active_model_id,
    infer_slot,
    load_model_and_tokenizer,
    loaded_paths,
    translate_glosses,
    translate_spanish_to_glosses,
)

app = FastAPI(title="LSA semántico", version="1.0.0")


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
    paths = loaded_paths()
    return {
        "ok": True,
        "model": get_active_model_id() or DEFAULT_MODEL_ID,
        "sordo": True,
        "oyente": True,
        "sordo_gguf": paths.get("sordo"),
        "oyente_gguf": paths.get("oyente"),
        "clients": n_clients(),
    }


def _key(request: Request, x_lsa_client: Optional[str]) -> str:
    peer = request.client.host if request.client else None
    return client_key(x_lsa_client, request.headers.get("x-forwarded-for"), peer)


@app.post("/sordo")
def sordo(
    body: SordoIn,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
    x_lsa_client: Optional[str] = Header(None, alias="X-LSA-Client"),
):
    _check_token(authorization, x_lsa_token)
    key = _key(request, x_lsa_client)
    with infer_slot() as queued_s:
        mem = memory_for(key)
        history = mem.as_messages() if USE_CONVERSATION_HISTORY else None
        t0 = time.perf_counter()
        spanish = translate_glosses(body.glosses, history_messages=history) or ""
        ms = (time.perf_counter() - t0) * 1000
        if spanish:
            mem.add_signer(spanish, glosses=body.glosses.strip())
        turns = len(mem.turns)
    return {
        "spanish": spanish,
        "glosses": body.glosses.strip(),
        "ms": round(ms, 1),
        "queued_ms": round(queued_s * 1000, 1),
        "client": key,
        "turns": turns,
    }


@app.post("/oyente")
def oyente(
    body: OyenteIn,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_lsa_token: Optional[str] = Header(None, alias="X-LSA-Token"),
    x_lsa_client: Optional[str] = Header(None, alias="X-LSA-Client"),
):
    _check_token(authorization, x_lsa_token)
    key = _key(request, x_lsa_client)
    with infer_slot() as queued_s:
        mem = memory_for(key)
        t0 = time.perf_counter()
        glosses = translate_spanish_to_glosses(body.spanish) or ""
        ms = (time.perf_counter() - t0) * 1000
        mem.add_hearing(body.spanish.strip())
        turns = len(mem.turns)
    return {
        "glosses": glosses,
        "spanish": body.spanish.strip(),
        "ms": round(ms, 1),
        "queued_ms": round(queued_s * 1000, 1),
        "client": key,
        "turns": turns,
    }


def warmup() -> None:
    print("[semantic-http] Cargando GGUF…")
    load_model_and_tokenizer()
    print("[semantic-http] Listo /sordo y /oyente")
