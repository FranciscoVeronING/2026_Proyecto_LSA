"""Cliente HTTP del servidor semántico en Cloud Run."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from semantic.config import CLOUD_SEMANTIC_URL, SEMANTIC_TOKEN

_TIMEOUT = 240.0
_KEEPALIVE_SEC = 240.0
_last_timing: dict = {
    "roundtrip_ms": None,
    "server_ms": None,
    "queued_ms": None,
    "infer_ms": None,
}
_keepalive_started = False
_keepalive_lock = threading.Lock()


def semantic_url() -> str:
    override = (os.getenv("LSA_SEMANTIC_URL") or "").strip().rstrip("/")
    return override or CLOUD_SEMANTIC_URL.rstrip("/")


def semantic_token() -> str:
    return (os.getenv("LSA_SEMANTIC_TOKEN") or SEMANTIC_TOKEN or "").strip()


def configured() -> bool:
    return bool(semantic_url())


def _client_id() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "client_id.txt"
    try:
        if path.is_file():
            got = path.read_text(encoding="utf-8").strip()
            if got:
                return got
        path.parent.mkdir(parents=True, exist_ok=True)
        got = uuid.uuid4().hex
        path.write_text(got + "\n", encoding="utf-8")
        return got
    except OSError:
        return "ilsa"


def _headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ILSA/1.0",
        "X-LSA-Client": _client_id(),
    }
    token = semantic_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-LSA-Token"] = token
    return headers


def _post(path: str, payload: dict, timeout: float = _TIMEOUT) -> dict:
    base = semantic_url()
    if not base:
        raise RuntimeError("Falta la URL del traductor semántico.")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers=_headers(),
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"semántico HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se alcanzó el semántico en {base}: {exc.reason}") from exc
    data = json.loads(body) if body else {}
    roundtrip = (time.perf_counter() - t0) * 1000
    infer_ms = data.get("ms")
    queued_ms = data.get("queued_ms")
    infer_val = float(infer_ms) if isinstance(infer_ms, (int, float)) else None
    queued_val = float(queued_ms) if isinstance(queued_ms, (int, float)) else None
    _last_timing["roundtrip_ms"] = round(roundtrip, 1)
    _last_timing["infer_ms"] = round(infer_val, 1) if infer_val is not None else None
    _last_timing["queued_ms"] = round(queued_val, 1) if queued_val is not None else None
    # `server_ms` = inferencia (RNF-01). La cola va en `queued_ms`.
    _last_timing["server_ms"] = _last_timing["infer_ms"]
    extra = ""
    if queued_val is not None:
        extra += f" cola={queued_val:.0f}ms"
    if infer_val is not None:
        extra += f" infer={infer_val:.0f}ms"
    print(f"[semantic] POST {path} roundtrip={roundtrip:.0f}ms{extra}")
    _ensure_keepalive()
    return data


def last_timing() -> dict:
    return dict(_last_timing)


def ping(timeout: float = _TIMEOUT) -> bool:
    base = semantic_url()
    if not base:
        print("[semantic] ping: no hay URL")
        return False
    req = urllib.request.Request(f"{base}/health", headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            print(f"[semantic] ping {base}/health → HTTP {resp.status}")
            if ok:
                _ensure_keepalive()
            return ok
    except Exception as exc:
        print(f"[semantic] ping {base}/health: {exc}")
        return False


def _ensure_keepalive() -> None:
    global _keepalive_started
    with _keepalive_lock:
        if _keepalive_started:
            return
        _keepalive_started = True

    def _loop() -> None:
        while True:
            time.sleep(_KEEPALIVE_SEC)
            try:
                ping(timeout=_TIMEOUT)
            except Exception:
                pass

    threading.Thread(target=_loop, name="semantic-keepalive", daemon=True).start()


def _first_str(data: dict, *keys: str) -> str:
    for key in keys:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def translate_sordo(glosses: str) -> str:
    """Cloud Run `POST /sign`: glosas → oración."""
    data = _post("/sign", {"glosses": glosses})
    return _first_str(data, "spanish", "text", "output")


def translate_oyente(spanish: str) -> str:
    """Cloud Run `POST /hearing`: oración → glosas."""
    data = _post("/hearing", {"spanish": spanish})
    return _first_str(data, "glosses", "caption", "text", "output")
