"""Cliente HTTP del servidor semántico (ngrok / Cloudflare Tunnel)."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from semantic.config import (
    SEMANTIC_TOKEN,
    SEMANTIC_URL,
    SEMANTIC_URL_ASSET,
    SEMANTIC_URL_REPO,
    SEMANTIC_URL_TAG,
)

_TIMEOUT = 60
_resolved: Optional[str] = None
_last_timing: dict = {"roundtrip_ms": None, "server_ms": None}


def _clean_url(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.rstrip("/")
    return ""


def _read_url_file(path: Path) -> str:
    try:
        if path.is_file():
            return _clean_url(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    return ""


def _cache_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    return Path(base) / "ILSA" / "semantic_url.txt"


def _exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def release_url_asset() -> str:
    override = os.getenv("LSA_SEMANTIC_URL_FILE", "").strip()
    if override:
        return override
    return (
        f"https://github.com/{SEMANTIC_URL_REPO}/releases/download/"
        f"{SEMANTIC_URL_TAG}/{SEMANTIC_URL_ASSET}"
    )


def _fetch_github() -> str:
    url = release_url_asset()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ILSA/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[semantic] No pude bajar {url}: {exc}")
        return ""
    got = _clean_url(body)
    if got:
        dest = _cache_path()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(got + "\n", encoding="utf-8")
        except OSError:
            pass
        print(f"[semantic] URL remota: {got}")
    return got


def semantic_url() -> str:
    """Env → GitHub Release semantic_url.txt → cache / archivo al lado del exe."""
    global _resolved
    if _resolved is not None:
        return _resolved

    env = (os.getenv("LSA_SEMANTIC_URL") or SEMANTIC_URL or "").strip().rstrip("/")
    if env:
        _resolved = env
        return _resolved

    fetched = _fetch_github()
    if fetched:
        _resolved = fetched
        return _resolved

    for path in (
        _exe_dir() / "semantic_url.txt",
        Path.cwd() / "semantic_url.txt",
        _cache_path(),
    ):
        local = _read_url_file(path)
        if local:
            _resolved = local
            print(f"[semantic] URL desde {path}")
            return _resolved

    _resolved = ""
    return _resolved


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
        import uuid

        got = uuid.uuid4().hex
        path.write_text(got + "\n", encoding="utf-8")
        return got
    except OSError:
        return "ilsa"


def _headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ILSA/1.0",
        "ngrok-skip-browser-warning": "1",
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
        raise RuntimeError(
            "No hay URL del semántico. Subí packaging/semantic_url.txt al release ilsa-llama-1b."
        )
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
    server_ms = data.get("ms")
    _last_timing["roundtrip_ms"] = round(roundtrip, 1)
    _last_timing["server_ms"] = float(server_ms) if isinstance(server_ms, (int, float)) else None
    return data


def last_timing() -> dict:
    return dict(_last_timing)


def ping(timeout: float = 8.0) -> bool:
    base = semantic_url()
    if not base:
        print("[semantic] ping: no hay URL")
        return False
    req = urllib.request.Request(f"{base}/health", headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        ok = bool(data.get("ok"))
        if not ok:
            print(f"[semantic] ping {base}/health: sin ok ({data!r})")
        return ok
    except Exception as exc:
        print(f"[semantic] ping {base}/health: {exc}")
        return False


def translate_sordo(glosses: str, history_messages: Optional[list[dict]] = None) -> str:
    """Misma firma que translator.translate_glosses (history se ignora en el POST)."""
    del history_messages
    data = _post("/sordo", {"glosses": glosses})
    return str(data.get("spanish") or "").strip()


def translate_oyente(spanish: str) -> str:
    data = _post("/oyente", {"spanish": spanish})
    return str(data.get("glosses") or "").strip()
