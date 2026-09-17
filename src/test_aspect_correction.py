"""
Testea la hipótesis del aspect ratio sobre los takes ya grabados, sin cámara.

La cámara (DroidCam libre) está capada a 640x480 y los videos de entrenamiento
son 1920x1080. MediaPipe divide x por el ancho e y por el alto, así que para un
mismo tamaño físico:

    vertical_final = (Δ/ancho_hombros) * (W/H)

con W/H = 1.778 al entrenar y 1.333 al capturar. Toda medida vertical le llega
al modelo a 0.75x de la escala que vio entrenando. Como el ancla es el punto
medio de hombros por eje y la escala es la distancia entre hombros (casi
horizontal), el efecto sobre el tensor ya normalizado es aislable: x queda
igual, y se multiplica por 4/3.

Esto permite testear la corrección sin volver a grabar: aplicarla a los takes
guardados y ver si las predicciones se arreglan. Es una aproximación —ignora la
inclinación de hombros y el eje z— así que un resultado positivo es evidencia
fuerte, pero uno negativo no cierra la hipótesis.

Uso (desde src/, entorno lsa_gpu):

    python test_aspect_correction.py eval_36_720p_takes
    python test_aspect_correction.py eval_36_720p_takes --model 2026_08_29_model_opt
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from camera import build_classifier, discover_models, resolve_model_root
from predict_npy import predict

TRAIN_ASPECT = 16 / 9
CAPTURE_ASPECT = 4 / 3


def scale_axis(seq: np.ndarray, axis: int, factor: float) -> np.ndarray:
    """Escala una coordenada (0=x, 1=y, 2=z) de un tensor (frames, 225)."""
    out = seq.astype(np.float32).copy()
    points = out.reshape(out.shape[0], -1, 3)
    # Los bloques de mano ausente son cero y deben seguir siendo cero; como
    # multiplicar por un factor preserva el cero, no hace falta enmascarar.
    points[:, :, axis] *= factor
    return points.reshape(out.shape[0], -1)


def take_label(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test offline de corrección de aspect ratio.")
    parser.add_argument("takes_dir", help="Directorio *_takes de camera.py --eval.")
    parser.add_argument("--model", default=None, help="Filtra por id de modelo.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    paths = sorted(glob.glob(os.path.join(args.takes_dir, "**", "*.npy"), recursive=True))
    if not paths:
        print(f"[!] Sin .npy en {args.takes_dir}.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = discover_models(resolve_model_root())
    if args.model:
        specs = [s for s in specs if args.model in s.id]
    if not specs:
        print("[!] Ningún modelo matcheó.")
        return 1

    factor = TRAIN_ASPECT / CAPTURE_ASPECT
    variants = [
        ("original", lambda s: s),
        (f"y x{factor:.3f}", lambda s: scale_axis(s, 1, factor)),
        (f"y,z x{factor:.3f}", lambda s: scale_axis(scale_axis(s, 1, factor), 2, factor)),
    ]

    print(f"Takes: {len(paths)} | factor de correccion: {factor:.4f}\n")

    for spec in specs:
        try:
            model = build_classifier(spec, device)
        except Exception as exc:
            print(f"[!] No se pudo cargar {spec.id}: {exc}")
            continue

        print(f"=== {spec.short} (max_frames={spec.max_frames}) ===")
        results: dict[str, list[tuple[str, str, float]]] = {}
        for name, fn in variants:
            rows = []
            for path in paths:
                expected = take_label(path)
                seq = np.load(path)
                if seq.shape[0] != spec.max_frames:
                    # El take se guardo con el nº de frames de otro modelo.
                    continue
                top3 = predict(spec, model, fn(seq), device)
                rows.append((expected, top3[0][0], top3[0][1]))
            results[name] = rows

        usable = len(results[variants[0][0]])
        if not usable:
            print(f"    (sin takes de {spec.max_frames} frames; se saltea)\n")
            continue

        for name, rows in results.items():
            hits = sum(1 for exp, got, _ in rows if exp.casefold() == got.casefold())
            print(f"    {name:14} top1 = {hits}/{len(rows)} = {hits / len(rows):.1%}")

        base = {exp: (got, c) for exp, got, c in results[variants[0][0]]}
        corr = {exp: (got, c) for exp, got, c in results[variants[1][0]]}
        print("\n    Cambios con la correccion de y:")
        changed = False
        for exp in sorted(base):
            b, c = base[exp], corr[exp]
            if b[0] == c[0]:
                continue
            changed = True
            ok_b = "OK" if exp.casefold() == b[0].casefold() else "  "
            ok_c = "OK" if exp.casefold() == c[0].casefold() else "  "
            print(f"      {exp:4} {ok_b} {b[0]:12}{b[1]:.3f}  ->  {ok_c} {c[0]:12}{c[1]:.3f}")
        if not changed:
            print("      (ninguna prediccion cambio)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
