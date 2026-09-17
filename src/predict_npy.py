"""
Referencia dorada: corre los modelos sobre un .npy ya extraído, sin cámara.

Para qué: la webapp falla más que esta branch, y hay dos mitades sospechosas.

    landmarks (MediaPipe)  →  normalización + trim + subsampleo  →  modelo

Este script fija la segunda mitad. Si la webapp, alimentada con el MISMO .npy,
da el mismo top-3 que acá, la divergencia está 100% en la extracción de
landmarks (versión de MediaPipe, resolución, fps, espejado). Si da distinto,
está en la carga del modelo o en el armado del tensor.

Uso (desde src/, entorno lsa_gpu):

    python predict_npy.py ../dataset_landmarks_16frames/I/I-1.npy
    python predict_npy.py --sign I --clips 3          # 3 clips de esa clase
    python predict_npy.py --sign I --models 2026_08_26_model_opt
    python predict_npy.py --sign I --clips 5 --json parity_I.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from camera import build_classifier, discover_models, resolve_model_root
from inspect_sign_features import _load_sequence, _npy_paths
from utils import normalize_sequence_to_frames


def pick_specs(wanted: list[str] | None) -> list:
    specs = discover_models(resolve_model_root())
    if not specs:
        raise FileNotFoundError("No encontré modelos con .pth en src/model.")
    if not wanted:
        return specs
    chosen = []
    for name in wanted:
        matches = [s for s in specs if name in s.id or name in s.label]
        if not matches:
            print(f"[!] Ningún modelo matcheó '{name}'.")
        chosen.extend(matches)
    return chosen or specs


def predict(spec, model, sequence: np.ndarray, device) -> list[tuple[str, float]]:
    """El .npy ya está trimeado y subsampleado; solo se ajusta el nº de frames."""
    matrix = normalize_sequence_to_frames(sequence, spec.max_frames)
    tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
    return [
        (spec.idx_to_class.get(int(i), "desconocido"), float(c))
        for c, i in zip(values.tolist(), indices.tolist())
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inferencia sobre .npy, sin cámara.")
    parser.add_argument("npy", nargs="*", help="Rutas .npy explícitas.")
    parser.add_argument("--sign", help="Clase del dataset (usa sus .npy).")
    parser.add_argument("--clips", type=int, default=1, help="Cuántos clips de --sign.")
    parser.add_argument("--models", nargs="*", help="Filtra modelos por id/label.")
    parser.add_argument("--json", help="Volcar resultados a JSON para diffear.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    paths = list(args.npy)
    if args.sign:
        found = _npy_paths(args.sign)
        if not found:
            print(f"[!] Sin .npy para '{args.sign}'.")
            return 1
        paths.extend(found[: args.clips])
    if not paths:
        print("[!] Pasá un .npy o --sign.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = pick_specs(args.models)
    print(f"Device: {device}  |  modelos: {len(specs)}\n")

    loaded = []
    for spec in specs:
        try:
            loaded.append((spec, build_classifier(spec, device)))
        except Exception as exc:
            print(f"[!] No se pudo cargar {spec.id}: {exc}")

    dump = []
    for path in paths:
        seq = _load_sequence(path)
        print(f"{os.path.basename(path)}   shape={seq.shape}")
        for spec, model in loaded:
            top3 = predict(spec, model, seq, device)
            pretty = " | ".join(f"{n.upper()} {c:.4f}" for n, c in top3)
            print(f"    {spec.short:<28} {pretty}")
            dump.append(
                {
                    "npy": os.path.abspath(path),
                    "model": spec.id,
                    "max_frames": spec.max_frames,
                    "top3": [{"label": n, "conf": round(c, 6)} for n, c in top3],
                }
            )
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(dump, handle, ensure_ascii=False, indent=2)
        print(f"Guardado {os.path.abspath(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
