"""Rutas estrictas de GGUF: solo ``src/semantic/models/{sordo,oyente}/``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from semantic.config import (
    BACKUP_GGUF,
    OYENTE_GGUF_DIR,
    PRIMARY_GGUF,
    SORDO_GGUF_DIR,
)


def gguf_candidates(folder: Path) -> list[Path]:
    """Principal (Qwen) y backup (Llama), luego cualquier otro ``*.gguf`` del folder."""
    out: list[Path] = []
    seen: set[Path] = set()
    for name in (PRIMARY_GGUF, BACKUP_GGUF):
        path = folder / name
        if path.is_file():
            resolved = path.resolve()
            out.append(resolved)
            seen.add(resolved)
    extras: list[Path] = []
    if folder.is_dir():
        for path in folder.glob("*.gguf"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            extras.append(resolved)
    extras.sort(key=lambda p: (0 if "qwen" in p.name.lower() else 1 if "llama" in p.name.lower() else 2, p.name.lower()))
    out.extend(extras)
    return out


def sordo_candidates() -> list[Path]:
    return gguf_candidates(SORDO_GGUF_DIR)


def oyente_candidates() -> list[Path]:
    return gguf_candidates(OYENTE_GGUF_DIR)


def describe_slot(folder: Path) -> dict:
    primary = folder / PRIMARY_GGUF
    backup = folder / BACKUP_GGUF
    return {
        "dir": str(folder.resolve()),
        "primary": PRIMARY_GGUF,
        "primary_present": primary.is_file(),
        "backup": BACKUP_GGUF,
        "backup_present": backup.is_file(),
    }


def model_used_name(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).name
