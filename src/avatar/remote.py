"""GET `/tposes/{GLOSS}.pose` from the server. Exact name match, no local dictionary."""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from avatar.config import AVATAR_TOKEN, CLOUD_AVATAR_URL
from avatar.names import expand_spell, tpose_stem

_TIMEOUT = 60.0
_MAX_SIGNS = 24
_TOKEN = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_DIGITS = {
    "0": "CERO",
    "1": "UNO",
    "2": "DOS",
    "3": "TRES",
    "4": "CUATRO",
    "5": "CINCO",
    "6": "SEIS",
    "7": "SIETE",
    "8": "OCHO",
    "9": "NUEVE",
}
_CACHE_LOCK = threading.Lock()
_tpose_stats: dict = {
    "tpose_gets": 0,
    "tpose_cache_hits": 0,
    "tpose_fetch_ms": 0.0,
    "tpose_bytes": 0,
}


def reset_tpose_stats() -> None:
    _tpose_stats.update(
        tpose_gets=0,
        tpose_cache_hits=0,
        tpose_fetch_ms=0.0,
        tpose_bytes=0,
    )


def tpose_stats() -> dict:
    return dict(_tpose_stats)


def avatar_url() -> str:
    override = (os.getenv("LSA_AVATAR_URL") or "").strip().rstrip("/")
    if override:
        return override
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "avatar_url.txt"
    try:
        if path.is_file():
            got = path.read_text(encoding="utf-8").strip().rstrip("/")
            if got:
                return got
    except OSError:
        pass
    return CLOUD_AVATAR_URL.rstrip("/")


def avatar_token() -> str:
    return (os.getenv("LSA_AVATAR_TOKEN") or AVATAR_TOKEN or "").strip()


def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "tposes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "ILSA/1.0"}
    token = avatar_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-LSA-Token"] = token
    return headers


def ping(timeout: float = 20.0) -> bool:
    """GET `/health` on the T-pose server."""
    base = avatar_url()
    if not base:
        print("[avatar] ping: no URL")
        return False
    req = urllib.request.Request(f"{base}/health", headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            print(f"[avatar] ping {base}/health → HTTP {resp.status}")
            return ok
    except Exception as exc:
        print(f"[avatar] ping {base}/health: {exc}")
        return False


def fetch_tpose(stem: str) -> Path | None:
    """GET `/tposes/{stem}.pose`. 404 → None. Cacha el archivo en disco."""
    key = (stem or "").strip()
    if not key:
        return None
    dest = _cache_dir() / f"{key}.pose"
    _tpose_stats["tpose_gets"] = int(_tpose_stats.get("tpose_gets") or 0) + 1
    if dest.is_file() and dest.stat().st_size > 64:
        _tpose_stats["tpose_cache_hits"] = int(_tpose_stats.get("tpose_cache_hits") or 0) + 1
        return dest
    url = f"{avatar_url()}/tposes/{key}.pose"
    req = urllib.request.Request(url, headers=_headers(), method="GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"T-pose HTTP {exc.code} for {key}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the T-pose server: {exc.reason}") from exc
    if not blob:
        return None
    tmp = dest.with_suffix(".pose.part")
    with _CACHE_LOCK:
        tmp.write_bytes(blob)
        tmp.replace(dest)
    ms = (time.perf_counter() - t0) * 1000
    _tpose_stats["tpose_fetch_ms"] = float(_tpose_stats.get("tpose_fetch_ms") or 0) + ms
    _tpose_stats["tpose_bytes"] = int(_tpose_stats.get("tpose_bytes") or 0) + len(blob)
    print(f"[avatar] GET /tposes/{key}.pose {len(blob)} bytes {ms:.0f}ms")
    return dest


def resolve_gloss_file(raw: str) -> Path | None:
    query = _DIGITS.get(raw, raw)
    stem = tpose_stem(query)
    if not stem:
        return None
    return fetch_tpose(stem)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text or "")


def resolve_chain(text: str) -> tuple[list[tuple[str, Path]], list[str]]:
    """Prueba de a 3, 2 y 1 token (BUENOS AIRES → BUENOSAIRES). Si no hay archivo, deletrea."""
    tokens = _tokens(text)
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    index = 0
    while index < len(tokens):
        if len(found) >= _MAX_SIGNS:
            missing.extend(tokens[index:])
            break
        matched = False
        for size in (3, 2, 1):
            if index + size > len(tokens):
                continue
            raw = " ".join(tokens[index : index + size])
            path = resolve_gloss_file(raw)
            if path is None:
                continue
            found.append((raw, path))
            index += size
            matched = True
            break
        if matched:
            continue
        query = _DIGITS.get(tokens[index], tokens[index])
        letters = expand_spell(query)
        if len(letters) < 2:
            missing.append(tokens[index])
            index += 1
            continue
        for letter in letters:
            if len(found) >= _MAX_SIGNS:
                missing.append(letter)
                continue
            path = resolve_gloss_file(letter)
            if path is None:
                missing.append(letter)
                continue
            found.append((letter, path))
        index += 1
    return found, missing


def clip_for_glosses(glosses: str) -> dict:
    """Hearing-semantic glosses → frames. Each sign is a GET to the T-pose server."""
    text = (glosses or "").strip()
    reset_tpose_stats()
    if not text:
        return {"ok": False, "error": "No glosses arrived.", "frames": [], "missing": []}
    t0 = time.perf_counter()
    try:
        found, missing = resolve_chain(text)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "frames": [], "missing": [], **tpose_stats()}
    resolve_ms = (time.perf_counter() - t0) * 1000
    if not found:
        return {
            "ok": False,
            "error": "No gloss has a T-pose on the server.",
            "frames": [],
            "missing": missing,
            "tpose_resolve_ms": round(resolve_ms, 1),
            **tpose_stats(),
        }
    from backend.ensure_pkg import ensure_module
    from poseclip.sequence import sequence_from_files

    ensure_module("pose_format", "pose-format")
    t_comp = time.perf_counter()
    clip = sequence_from_files(found, missing)
    compose_ms = (time.perf_counter() - t_comp) * 1000
    clip["tpose_resolve_ms"] = round(resolve_ms, 1)
    clip["tpose_compose_ms"] = round(compose_ms, 1)
    clip["tpose_missing_n"] = len(clip.get("missing") or [])
    clip.update(tpose_stats())
    return clip
