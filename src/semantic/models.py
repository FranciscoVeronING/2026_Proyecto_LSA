"""Catálogo de modelos semánticos (GGUF) en src/semantic/outputs/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from semantic.config import OUTPUTS_DIR


@dataclass(frozen=True)
class SemanticModelSpec:
    id: str
    folder: str
    chat_format: str  # chatml | llama3


# Etiquetas de UI ↔ carpetas Unsloth exportadas a GGUF.
SEMANTIC_MODEL_SPECS: tuple[SemanticModelSpec, ...] = (
    SemanticModelSpec("qwen2.5-0.5b", "unsloth_Qwen2.5-0.5B-Instruct", "chatml"),
    SemanticModelSpec("qwen2.5-1.5b", "unsloth_Qwen2.5-1.5B-Instruct", "chatml"),
    SemanticModelSpec("qwen2.5-3b", "unsloth_Qwen2.5-3B-Instruct", "chatml"),
    SemanticModelSpec("llama-3.2-1b", "unsloth_Llama-3.2-1B-Instruct", "llama3"),
    SemanticModelSpec("smollm2-1.7b", "unsloth_SmolLM2-1.7B-Instruct", "chatml"),
)


def spec_by_id(model_id: str) -> SemanticModelSpec:
    for spec in SEMANTIC_MODEL_SPECS:
        if spec.id == model_id:
            return spec
    known = ", ".join(s.id for s in SEMANTIC_MODEL_SPECS)
    raise KeyError(f"Modelo semántico desconocido: {model_id!r}. Opciones: {known}")


def resolve_gguf_path(spec: SemanticModelSpec) -> Optional[Path]:
    """Busca el .gguf en `{folder}_gguf/` o en `{folder}/`."""
    candidates = (
        OUTPUTS_DIR / f"{spec.folder}_gguf",
        OUTPUTS_DIR / spec.folder,
    )
    for folder in candidates:
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.gguf"))
        if files:
            return files[0]
    return None


def list_semantic_models() -> List[dict]:
    """Opciones para el dropdown: id, label, available, path."""
    items = []
    for spec in SEMANTIC_MODEL_SPECS:
        path = resolve_gguf_path(spec)
        items.append(
            {
                "id": spec.id,
                "label": spec.id,
                "available": path is not None,
                "path": str(path) if path else None,
                "chat_format": spec.chat_format,
            }
        )
    return items
