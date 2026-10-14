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
    """Principal (Qwen) y backup (Llama), en ese orden, si el archivo existe."""
    out: list[Path] = []
    for name in (PRIMARY_GGUF, BACKUP_GGUF):
        path = folder / name
        if path.is_file():
            out.append(path.resolve())
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
