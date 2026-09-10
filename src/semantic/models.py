"""Los tres GGUF que lista ILSA. Sin archivo en outputs/ → available=False."""

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


# Nombres para la UI (sin ids técnicos) y cuánto cuestan de correr.
SEMANTIC_FRIENDLY_LABELS: dict[str, str] = {
    "qwen2.5-0.5b": "Traducción rápida",
    "qwen2.5-3b": "Traducción precisa",
    "llama-3.2-1b": "Traducción compacta",
}

# liviano < medio < pesado (tamaño del modelo / CPU-RAM al traducir).
SEMANTIC_COMPUTE: dict[str, str] = {
    "qwen2.5-0.5b": "liviano",
    "qwen2.5-3b": "pesado",
    "llama-3.2-1b": "liviano",
}


def friendly_label(model_id: str) -> str:
    if not model_id:
        return SEMANTIC_FRIENDLY_LABELS["qwen2.5-3b"]
    return SEMANTIC_FRIENDLY_LABELS.get(model_id, "Traducción")


def compute_label(model_id: str) -> str:
    key = model_id or "qwen2.5-3b"
    return SEMANTIC_COMPUTE.get(key, "medio")


def display_label(model_id: str) -> str:
    return f"{friendly_label(model_id)} · {compute_label(model_id)}"


# Etiquetas de UI ↔ carpetas Unsloth exportadas a GGUF.
SEMANTIC_MODEL_SPECS: tuple[SemanticModelSpec, ...] = (
    SemanticModelSpec("qwen2.5-0.5b", "unsloth_Qwen2.5-0.5B-Instruct", "chatml"),
    SemanticModelSpec("qwen2.5-3b", "unsloth_Qwen2.5-3B-Instruct", "chatml"),
    SemanticModelSpec("llama-3.2-1b", "unsloth_Llama-3.2-1B-Instruct", "llama3"),
)


def spec_by_id(model_id: str) -> SemanticModelSpec:
    for spec in SEMANTIC_MODEL_SPECS:
        if spec.id == model_id:
            return spec
    known = ", ".join(s.id for s in SEMANTIC_MODEL_SPECS)
    raise KeyError(f"Modelo semántico desconocido: {model_id!r}. Opciones: {known}")


def resolve_gguf_path(spec: SemanticModelSpec) -> Optional[Path]:
    # Unsloth a veces deja el .gguf en folder_gguf, a veces en folder.
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
