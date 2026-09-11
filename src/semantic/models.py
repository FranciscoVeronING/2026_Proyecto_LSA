"""Un solo GGUF: Llama 3.2 1B. Sin archivo local → available=False hasta descargarlo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import os

from semantic.config import OUTPUTS_DIR


@dataclass(frozen=True)
class SemanticModelSpec:
    id: str
    folder: str
    chat_format: str  # chatml | llama3


GGUF_ASSET_NAME = "llama-3.2-1b-instruct.Q4_K_M.gguf"

SEMANTIC_FRIENDLY_LABELS: dict[str, str] = {
    "llama-3.2-1b": "Traductor",
}

SEMANTIC_COMPUTE: dict[str, str] = {
    "llama-3.2-1b": "liviano",
}

SEMANTIC_MODEL_SPECS: tuple[SemanticModelSpec, ...] = (
    SemanticModelSpec("llama-3.2-1b", "unsloth_Llama-3.2-1B-Instruct", "llama3"),
)


def friendly_label(model_id: str) -> str:
    return SEMANTIC_FRIENDLY_LABELS.get(model_id or "llama-3.2-1b", "Traductor")


def compute_label(model_id: str) -> str:
    return SEMANTIC_COMPUTE.get(model_id or "llama-3.2-1b", "liviano")


def display_label(model_id: str) -> str:
    return friendly_label(model_id)


def spec_by_id(model_id: str) -> SemanticModelSpec:
    for spec in SEMANTIC_MODEL_SPECS:
        if spec.id == model_id:
            return spec
    raise KeyError(f"Modelo semántico desconocido: {model_id!r}")


def resolve_gguf_path(spec: SemanticModelSpec) -> Optional[Path]:
    appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    candidates = (
        OUTPUTS_DIR / f"{spec.folder}_gguf" / GGUF_ASSET_NAME,
        OUTPUTS_DIR / spec.folder / GGUF_ASSET_NAME,
        Path(appdata) / "ILSA" / "models" / GGUF_ASSET_NAME,
    )
    for path in candidates:
        if path.is_file():
            return path
    for folder in (
        OUTPUTS_DIR / f"{spec.folder}_gguf",
        OUTPUTS_DIR / spec.folder,
    ):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.gguf"))
        if files:
            return files[0]
    return None


def list_semantic_models() -> List[dict]:
    items = []
    for spec in SEMANTIC_MODEL_SPECS:
        path = resolve_gguf_path(spec)
        items.append(
            {
                "id": spec.id,
                "label": display_label(spec.id),
                "available": path is not None,
                "path": str(path) if path else None,
                "chat_format": spec.chat_format,
            }
        )
    return items
