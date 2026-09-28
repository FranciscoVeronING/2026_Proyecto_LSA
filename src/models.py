"""Carga de checkpoints TinySkeleton (solo `src/model/*.pth`, sin archivados)."""
from __future__ import annotations

import json
import os

import torch

import config as cfg
from model_arch import TinySkeletonClassifier


def resolve_model_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "model")
    if os.path.isdir(candidate):
        return candidate
    return os.path.normpath(os.path.join(here, cfg.MODEL_SAVE_DIR))


def _safe_json(path: str):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def infer_arch_from_checkpoint(pth_path: str) -> dict:
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    hidden = int(state["conv_extractor.0.weight"].shape[0])
    n_cls = int(state["classification_head.weight"].shape[0])
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("transformer.layers.")
    }
    n_layers = (max(layer_ids) + 1) if layer_ids else cfg.NUM_LAYERS
    return {
        "hidden_dim": hidden,
        "num_layers": n_layers,
        "num_classes": n_cls,
        "compatible": "attention_pool.weight" in state,
    }


def _load_class_mapping(folder: str, metrics: dict, model_root: str, num_classes: int) -> dict:
    mapeo = _safe_json(os.path.join(folder, "mapeo_clases.json"))
    if isinstance(mapeo, dict) and len(mapeo) == num_classes:
        return {str(k): int(v) for k, v in mapeo.items()}
    classes = metrics.get("classes") if metrics else None
    if isinstance(classes, list) and len(classes) == num_classes:
        return {name: idx for idx, name in enumerate(classes)}
    mapeo = _safe_json(os.path.join(model_root, "mapeo_clases.json"))
    if isinstance(mapeo, dict) and len(mapeo) == num_classes:
        return {str(k): int(v) for k, v in mapeo.items()}
    return {name: idx for idx, name in enumerate(cfg.SIGN_CLASSES[:num_classes])}


class ModelSpec:
    def __init__(
        self,
        spec_id,
        short,
        label,
        folder,
        pth_path,
        hidden_dim,
        num_heads,
        num_layers,
        dropout_rate,
        max_frames,
        class_to_idx,
        archived=False,
        num_classes=None,
        val_acc=None,
    ):
        self.id = spec_id
        self.short = short
        self.label = label
        self.folder = folder
        self.pth_path = pth_path
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_frames = max_frames
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = int(num_classes) if num_classes is not None else len(class_to_idx)
        self.archived = archived
        self.val_acc = val_acc


def discover_models(model_root: str) -> list[ModelSpec]:
    """Un spec por `.pth` en la raíz de `src/model/` (no baja a subcarpetas)."""
    specs = []
    if not os.path.isdir(model_root):
        return specs

    pth_files = [
        name for name in os.listdir(model_root) if name.lower().endswith(".pth")
    ]
    preferred = [name for name in pth_files if name == cfg.MODEL_CHECKPOINT]
    names = preferred or sorted(pth_files)
    metrics = _safe_json(os.path.join(model_root, "metrics.json")) or {}

    seen = set()
    for pth_name in names:
        if pth_name in seen:
            continue
        seen.add(pth_name)
        pth_path = os.path.join(model_root, pth_name)
        spec_id = os.path.splitext(pth_name)[0]
        try:
            ckpt = infer_arch_from_checkpoint(pth_path)
        except Exception as exc:
            print(f"[!] No se pudo leer {pth_path}: {exc}")
            continue
        if not ckpt.get("compatible", True):
            print(f"[!] Omitido (sin attention pool): {spec_id}")
            continue

        hidden_dim = ckpt["hidden_dim"]
        num_layers = ckpt["num_layers"]
        metrics_ok = int(metrics.get("hidden_dim", -1)) == hidden_dim
        num_heads = int(metrics["num_heads"]) if metrics_ok and metrics.get("num_heads") else (
            4 if hidden_dim % 4 == 0 and hidden_dim <= 128 else (2 if hidden_dim % 2 == 0 else 1)
        )
        dropout_rate = float(
            metrics["dropout_rate"] if metrics_ok and metrics.get("dropout_rate") is not None else cfg.DROPOUT_RATE
        )
        max_frames = int(
            metrics["max_frames"] if metrics_ok and metrics.get("max_frames") is not None else cfg.MAX_FRAMES
        )
        class_to_idx = _load_class_mapping(model_root, metrics, model_root, ckpt["num_classes"])
        parts = [f"{ckpt['num_classes']}c", f"{max_frames}f", f"{hidden_dim}d", f"{num_heads}H{num_layers}L"]
        acc = metrics.get("val_accuracy_top1_pct") if metrics_ok else None
        if acc is not None:
            parts.append(f"val {acc:.0f}%")
        label = f"{spec_id} | {' '.join(parts)}"
        specs.append(
            ModelSpec(
                spec_id=spec_id,
                short=spec_id,
                label=label,
                folder=model_root,
                pth_path=pth_path,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=float(dropout_rate),
                max_frames=max_frames,
                class_to_idx=class_to_idx,
                archived=False,
                num_classes=ckpt["num_classes"],
                val_acc=acc,
            )
        )
    return specs


def default_model_index(catalog: list[ModelSpec]) -> int:
    for i, spec in enumerate(catalog):
        if os.path.basename(spec.pth_path) == cfg.MODEL_CHECKPOINT:
            return i
    return 0 if catalog else -1


def build_classifier(spec: ModelSpec, device):
    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM,
        spec.hidden_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        num_classes=spec.num_classes,
        dropout_rate=spec.dropout_rate,
    ).to(device)
    state = torch.load(spec.pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
