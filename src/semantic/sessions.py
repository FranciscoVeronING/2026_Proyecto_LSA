"""Una ventana de turnos por cliente (IP / id de ILSA). El GGUF no tiene memoria propia."""

from __future__ import annotations

import time
from threading import Lock
from typing import Optional

from core.conversation_memory import ConversationMemory
from semantic.config import CONVERSATION_HISTORY_SIZE

_MAX_CLIENTS = 80
_IDLE_SEC = 45 * 60

_lock = Lock()
_by_key: dict[str, ConversationMemory] = {}
_last_used: dict[str, float] = {}


def client_key(x_lsa_client: Optional[str], forwarded_for: Optional[str], peer: Optional[str]) -> str:
    cid = (x_lsa_client or "").strip()[:80]
    ip = ""
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    if not ip:
        ip = (peer or "").strip()
    if cid and ip:
        return f"{ip}|{cid}"
    return cid or ip or "unknown"


def _prune(now: float) -> None:
    stale = [k for k, ts in _last_used.items() if now - ts > _IDLE_SEC]
    for k in stale:
        _by_key.pop(k, None)
        _last_used.pop(k, None)
    if len(_by_key) <= _MAX_CLIENTS:
        return
    ordered = sorted(_last_used, key=_last_used.get)
    for k in ordered[: len(_by_key) - _MAX_CLIENTS]:
        _by_key.pop(k, None)
        _last_used.pop(k, None)


def memory_for(key: str) -> ConversationMemory:
    now = time.time()
    with _lock:
        _prune(now)
        mem = _by_key.get(key)
        if mem is None:
            mem = ConversationMemory(maxlen=CONVERSATION_HISTORY_SIZE)
            _by_key[key] = mem
        _last_used[key] = now
        return mem


def n_clients() -> int:
    with _lock:
        return len(_by_key)
