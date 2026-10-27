"""On-disk GGUF inventory. Does not load weights."""

from __future__ import annotations

from pathlib import Path

from semantic.config import BACKUP_GGUF, PRIMARY_GGUF


def list_ggufs(folder: Path) -> list[Path]:
    """Primary, backup, then any other ``*.gguf`` in the folder."""
    found: list[Path] = []
    seen: set[Path] = set()
    for name in (PRIMARY_GGUF, BACKUP_GGUF):
        path = folder / name
        if path.is_file():
            resolved = path.resolve()
            found.append(resolved)
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
    extras.sort(
        key=lambda p: (
            0 if "qwen" in p.name.lower() else 1 if "llama" in p.name.lower() else 2,
            p.name.lower(),
        )
    )
    found.extend(extras)
    return found


def disk_inventory(folder: Path) -> dict:
    return {
        "dir": str(folder.resolve()),
        "primary": PRIMARY_GGUF,
        "primary_present": (folder / PRIMARY_GGUF).is_file(),
        "backup": BACKUP_GGUF,
        "backup_present": (folder / BACKUP_GGUF).is_file(),
    }
