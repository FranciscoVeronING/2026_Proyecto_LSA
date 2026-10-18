"""Pide `/tposes/{GLOSA}.pose` al servidor. Igualdad de nombre, sin diccionario local."""

from __future__ import annotations

import os
import re
import sys
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


def fetch_tpose(stem: str) -> Path | None:
    """GET `/tposes/{stem}.pose`. 404 → None. Cacha el archivo en disco."""
    key = (stem or "").strip()
    if not key:
        return None
    dest = _cache_dir() / f"{key}.pose"
    if dest.is_file() and dest.stat().st_size > 64:
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
        raise RuntimeError(f"T-pose HTTP {exc.code} para {key}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se alcanzó el servidor de T-pose: {exc.reason}") from exc
    if not blob:
        return None
    tmp = dest.with_suffix(".pose.part")
    with _CACHE_LOCK:
        tmp.write_bytes(blob)
        tmp.replace(dest)
    ms = (time.perf_counter() - t0) * 1000
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
    """Glosas del semántico oyente → frames. Cada seña es GET al servidor de T-pose."""
    text = (glosses or "").strip()
    if not text:
        return {"ok": False, "error": "No llegaron glosas.", "frames": [], "missing": []}
    try:
        found, missing = resolve_chain(text)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "frames": [], "missing": []}
    if not found:
        return {
            "ok": False,
            "error": "Ninguna glosa tiene T-pose en el servidor.",
            "frames": [],
            "missing": missing,
        }
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from extraction.meet_sequence import sequence_from_files

    return sequence_from_files(found, missing)
