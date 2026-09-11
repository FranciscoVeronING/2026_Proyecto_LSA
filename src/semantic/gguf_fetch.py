"""Baja el GGUF de Llama 1B desde GitHub Releases si no está en disco."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from semantic.config import OUTPUTS_DIR
from semantic.models import GGUF_ASSET_NAME, spec_by_id, resolve_gguf_path
from semantic.config import DEFAULT_MODEL_ID

ProgressFn = Callable[[int, int, str], None]

GITHUB_REPO = os.environ.get(
    "LSA_GGUF_REPO", "FranciscoVeronING/2026_Proyecto_LSA"
)
RELEASE_TAG = os.environ.get("LSA_GGUF_TAG", "ilsa-llama-1b")
MIN_BYTES = 80 * 1024 * 1024


def release_asset_url() -> str:
    override = os.environ.get("LSA_GGUF_URL", "").strip()
    if override:
        return override
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{RELEASE_TAG}/{GGUF_ASSET_NAME}"
    )


def user_model_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    return Path(base) / "ILSA" / "models"


def cached_gguf_path() -> Path:
    return user_model_dir() / GGUF_ASSET_NAME


def _looks_like_gguf(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < MIN_BYTES:
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def find_local_gguf() -> Optional[Path]:
    spec = spec_by_id(DEFAULT_MODEL_ID)
    existing = resolve_gguf_path(spec)
    if existing and _looks_like_gguf(existing):
        return existing
    cached = cached_gguf_path()
    if _looks_like_gguf(cached):
        return cached
    if getattr(sys, "frozen", False):
        beside = Path(sys.executable).resolve().parent / "models" / GGUF_ASSET_NAME
        if _looks_like_gguf(beside):
            return beside
    return None


def _report(progress: Optional[ProgressFn], done: int, total: int, msg: str) -> None:
    if progress:
        progress(done, total, msg)


def ensure_gguf(progress: Optional[ProgressFn] = None) -> Path:
    """Devuelve la ruta del GGUF. Descarga si hace falta."""
    found = find_local_gguf()
    if found is not None:
        _report(progress, 1, 1, "Traductor listo")
        return found

    dest = cached_gguf_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = release_asset_url()
    _report(progress, 0, 0, "Descargando el traductor…")
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ILSA/1.0 (Llama-3.2-1B)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = min(99, int(done * 100 / total))
                        _report(progress, done, total, f"Descargando el traductor… {pct}%")
                    else:
                        mb = done / (1024 * 1024)
                        _report(progress, done, 0, f"Descargando el traductor… {mb:.0f} MB")
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise FileNotFoundError(
                "No está el modelo en GitHub Releases. "
                "Hay que publicarlo con packaging/upload_llama_gguf.ps1 "
                f"(tag {RELEASE_TAG})."
            ) from exc
        raise RuntimeError(f"No se pudo descargar el traductor ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Sin conexión o GitHub no responde. Revisá internet y reintentá."
        ) from exc

    if not _looks_like_gguf(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("La descarga no es un modelo válido. Reintentá.")
    tmp.replace(dest)
    # En desarrollo también queda visible junto a los otros GGUF.
    if not getattr(sys, "frozen", False):
        mirror = OUTPUTS_DIR / "unsloth_Llama-3.2-1B-Instruct_gguf" / GGUF_ASSET_NAME
        if not _looks_like_gguf(mirror):
            try:
                mirror.parent.mkdir(parents=True, exist_ok=True)
                if not mirror.exists():
                    os.link(dest, mirror)
            except OSError:
                pass
    _report(progress, 1, 1, "Traductor listo")
    return dest
