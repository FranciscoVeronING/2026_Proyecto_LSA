"""
Análisis de confusiones entre señas sobre todo el dataset .npy.

Dos niveles complementarios:

  A) Espacio crudo (T, 225) aplanado = lo que realmente ve TinySkeleton.
     Un 1-NN leave-one-out da una matriz de confusión y responde
     "¿qué pares están geométricamente pegados?".

  B) Las 12 features interpretables (6 mano-rostro + 6 movimiento de muñeca).
     Para cada par, el AUC por feature dice CUÁL feature podría separarlas.
     Es el puente entre "se confunden" y "qué habría que medir".

El nivel A predice confusión; el nivel B la explica. Ninguno reemplaza a
camera.py --eval: acá no hay luz mala ni MediaPipe inestable.

Uso (desde src/, entorno lsa_gpu):

    python analyze_sign_confusions.py
    python analyze_sign_confusions.py --pairs "L,Lunes" "T,I" --no-global
    python analyze_sign_confusions.py --top-pairs 40
"""

from __future__ import annotations

import argparse
import csv
import os
import unicodedata
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hand_face_features import (
    FEATURE_NAMES,
    HANDSHAPE_NAMES,
    MOTION_NAMES,
    clip_handshape_row,
    clip_motion_stats,
    clip_summary_row,
    hand_face_features_sequence,
)
from inspect_sign_features import _load_sequence, _npy_paths, available_signs

BASE_FEATURES = tuple(FEATURE_NAMES) + tuple(MOTION_NAMES)
ALL_FEATURES = BASE_FEATURES + tuple(HANDSHAPE_NAMES)

# Pedidos explícitamente. Los sueltos se reportan por vecinos más cercanos.
REQUESTED_PAIRS = [
    ("L", "Lunes"),
    ("Martes", "Chau"),
    ("F", "Donde"),
    ("Años", "G"),
    ("I", "Ojo"),
    ("Como", "Poder"),
    ("0", "O"),
    ("T", "I"),
]
REQUESTED_SINGLES = ["Nombre", "D", "vos", "Nosotros", "Ellos", "Lugar", "Ayer", "Papa"]

FEATURE_MEANING = {
    "dy_index_mouth": "altura índice-boca (<0 = arriba de la boca)",
    "dy_index_eye": "altura índice-ojo",
    "dy_index_nose": "altura índice-nariz",
    "dist_index_mouth": "distancia XY índice-boca",
    "dist_index_eye": "distancia XY índice-ojo",
    "dy_wrist_nose": "altura muñeca-nariz (pecho vs mentón)",
    "std_wrist_x": "agite lateral de la muñeca",
    "std_wrist_y": "agite vertical de la muñeca",
    "std_wrist_z": "agite en profundidad de la muñeca",
    "range_wrist_x": "amplitud lateral",
    "range_wrist_y": "amplitud vertical",
    "range_wrist_z": "amplitud en profundidad",
}
for _finger in ("thumb", "index", "middle", "ring", "pinky"):
    FEATURE_MEANING[f"ext_{_finger}"] = f"extensión del {_finger} (punta-muñeca)"
    FEATURE_MEANING[f"curl_{_finger}"] = f"curvatura del {_finger} (1 = recto)"
for _a, _b in (("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
    FEATURE_MEANING[f"spread_{_a}_{_b}"] = f"apertura entre {_a} y {_b}"


def _fold(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold().replace(" ", "_")


def resolve_name(wanted: str, available: list[str]) -> str | None:
    target = _fold(wanted)
    for name in available:
        if _fold(name) == target:
            return name
    return None


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def load_dataset(max_clips: int | None) -> dict:
    signs = available_signs()
    if not signs:
        raise FileNotFoundError("No hay .npy. Corré preprocessing.py primero.")

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    geo_rows: list[np.ndarray] = []
    geo_labels: list[int] = []
    skipped = 0

    for idx, sign in enumerate(signs):
        paths = _npy_paths(sign)
        if max_clips is not None:
            paths = paths[:max_clips]
        for path in paths:
            seq = _load_sequence(path)
            feats = hand_face_features_sequence(seq)
            summary = clip_summary_row(feats)
            motion = clip_motion_stats(seq)
            shape = clip_handshape_row(seq)
            sequences.append(seq)
            labels.append(idx)
            names.append(os.path.basename(path))
            if summary is None or motion is None or shape is None:
                skipped += 1
                continue
            geo_rows.append(np.concatenate([summary, motion, shape]))
            geo_labels.append(idx)
        print(f"  [{idx + 1:>3}/{len(signs)}] {sign:<14} {len(paths):>3} clips", flush=True)

    shapes = {seq.shape for seq in sequences}
    if len(shapes) != 1:
        raise ValueError(f"Clips con shapes distintos: {shapes}")

    return {
        "signs": signs,
        "raw": np.stack(sequences).reshape(len(sequences), -1),
        "raw_labels": np.asarray(labels, dtype=np.int32),
        "clip_names": names,
        "geo": np.stack(geo_rows).astype(np.float32),
        "geo_labels": np.asarray(geo_labels, dtype=np.int32),
        "skipped": skipped,
    }


# --------------------------------------------------------------------------
# Nivel A: 1-NN leave-one-out sobre el espacio crudo
# --------------------------------------------------------------------------


def loo_1nn(matrix: np.ndarray, labels: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Vecino más cercano de cada clip, excluyéndose a sí mismo."""
    sq = np.einsum("ij,ij->i", matrix, matrix)
    nearest = np.empty(matrix.shape[0], dtype=np.int32)
    for start in range(0, matrix.shape[0], chunk):
        stop = min(start + chunk, matrix.shape[0])
        block = matrix[start:stop]
        d = sq[None, :] - 2.0 * (block @ matrix.T) + np.einsum("ij,ij->i", block, block)[:, None]
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.argmin(d, axis=1)
    return labels[nearest]


def confusion_pairs(labels: np.ndarray, predicted: np.ndarray, signs: list[str]) -> list[dict]:
    """Pares (a, b) con cuántos clips de a caen en b y viceversa, simétrico."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    per_class_total = np.bincount(labels, minlength=len(signs))
    for true_idx, pred_idx in zip(labels, predicted):
        if true_idx == pred_idx:
            continue
        key = (min(true_idx, pred_idx), max(true_idx, pred_idx))
        counts[key] += 1

    rows = []
    for (a, b), n in counts.items():
        denom = per_class_total[a] + per_class_total[b]
        rows.append(
            {
                "a": signs[a],
                "b": signs[b],
                "n_confused": n,
                "rate": n / denom if denom else 0.0,
            }
        )
    rows.sort(key=lambda r: r["n_confused"], reverse=True)
    return rows


def per_sign_accuracy(labels: np.ndarray, predicted: np.ndarray, signs: list[str]) -> list[dict]:
    rows = []
    for idx, sign in enumerate(signs):
        mask = labels == idx
        n = int(np.count_nonzero(mask))
        if not n:
            continue
        hits = int(np.count_nonzero(predicted[mask] == idx))
        wrong = predicted[mask][predicted[mask] != idx]
        top_attractor = "-"
        if wrong.size:
            vals, cnt = np.unique(wrong, return_counts=True)
            top_attractor = signs[int(vals[int(np.argmax(cnt))])]
        rows.append(
            {
                "sign": sign,
                "n_clips": n,
                "acc": hits / n,
                "top_attractor": top_attractor,
            }
        )
    rows.sort(key=lambda r: r["acc"])
    return rows


# --------------------------------------------------------------------------
# Nivel B: separabilidad por feature
# --------------------------------------------------------------------------


def feature_auc(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """AUC de Mann-Whitney. 0.5 = indistinguible, 1.0 = separación perfecta."""
    n_a, n_b = values_a.size, values_b.size
    if not n_a or not n_b:
        return 0.5
    joined = np.concatenate([values_a, values_b])
    ranks = joined.argsort().argsort().astype(np.float64) + 1.0
    # Promedio de rangos para empates.
    order = np.argsort(joined, kind="mergesort")
    sorted_vals = joined[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    rank_sum_a = ranks[:n_a].sum()
    return float((rank_sum_a - n_a * (n_a + 1) / 2.0) / (n_a * n_b))


def pair_combined_acc(data: dict, sign_a: str, sign_b: str, features: tuple[str, ...]) -> float:
    """
    1-NN leave-one-out entre dos clases usando solo ese subconjunto de features.

    Responde lo que una feature sola no puede: ¿el conjunto las separa en
    combinación? 0.5 = azar, 1.0 = perfecto.
    """
    idx_a = data["signs"].index(sign_a)
    idx_b = data["signs"].index(sign_b)
    cols = [ALL_FEATURES.index(name) for name in features]
    mask = (data["geo_labels"] == idx_a) | (data["geo_labels"] == idx_b)
    x = data["geo"][mask][:, cols].astype(np.float64)
    y = (data["geo_labels"][mask] == idx_b).astype(np.int32)
    if x.shape[0] < 4:
        return 0.5
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    x = (x - x.mean(axis=0)) / std
    sq = np.einsum("ij,ij->i", x, x)
    d = sq[None, :] - 2.0 * (x @ x.T) + sq[:, None]
    np.fill_diagonal(d, np.inf)
    return float(np.mean(y[np.argmin(d, axis=1)] == y))


def pair_feature_table(
    data: dict, sign_a: str, sign_b: str, features: tuple[str, ...] = ALL_FEATURES
) -> list[dict]:
    idx_a = data["signs"].index(sign_a)
    idx_b = data["signs"].index(sign_b)
    mask_a = data["geo_labels"] == idx_a
    mask_b = data["geo_labels"] == idx_b
    rows = []
    for name in features:
        col = ALL_FEATURES.index(name)
        va = data["geo"][mask_a, col]
        vb = data["geo"][mask_b, col]
        auc = feature_auc(va, vb)
        rows.append(
            {
                "feature": name,
                "mean_a": float(va.mean()),
                "std_a": float(va.std()),
                "mean_b": float(vb.mean()),
                "std_b": float(vb.std()),
                "auc": auc,
                "sep": abs(auc - 0.5) * 2.0,
            }
        )
    rows.sort(key=lambda r: r["sep"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------


def plot_pair(data: dict, sign_a: str, sign_b: str, table: list[dict], out_path: str) -> None:
    idx_a = data["signs"].index(sign_a)
    idx_b = data["signs"].index(sign_b)
    mask_a = data["geo_labels"] == idx_a
    mask_b = data["geo_labels"] == idx_b

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"{sign_a}  vs  {sign_b}   —   mejor separador: {table[0]['feature']} "
        f"(AUC {table[0]['auc']:.2f}, sep {table[0]['sep']:.2f})",
        fontsize=12,
    )

    # Fila 1: histogramas de las 4 features que mejor separan.
    for pos, row in enumerate(table[:4]):
        ax = fig.add_subplot(3, 4, pos + 1)
        col = ALL_FEATURES.index(row["feature"])
        va = data["geo"][mask_a, col]
        vb = data["geo"][mask_b, col]
        bins = np.histogram_bin_edges(np.concatenate([va, vb]), bins=18)
        ax.hist(va, bins=bins, alpha=0.55, label=sign_a, color="C0", density=True)
        ax.hist(vb, bins=bins, alpha=0.55, label=sign_b, color="C3", density=True)
        ax.set_title(f"{row['feature']}\nsep={row['sep']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if pos == 0:
            ax.legend(fontsize=7)

    # Fila 2: scatter de las 2 mejores + las 2 peores (control).
    ax = fig.add_subplot(3, 4, 5)
    cx = ALL_FEATURES.index(table[0]["feature"])
    cy = ALL_FEATURES.index(table[1]["feature"])
    ax.scatter(data["geo"][mask_a, cx], data["geo"][mask_a, cy], s=16, alpha=0.6, label=sign_a, color="C0")
    ax.scatter(data["geo"][mask_b, cx], data["geo"][mask_b, cy], s=16, alpha=0.6, label=sign_b, color="C3")
    ax.set_xlabel(table[0]["feature"], fontsize=8)
    ax.set_ylabel(table[1]["feature"], fontsize=8)
    ax.set_title("2 mejores features", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # Fila 2 (resto): trayectoria media de la muñeca por eje.
    traj_a = mean_wrist_trajectory(data, idx_a)
    traj_b = mean_wrist_trajectory(data, idx_b)
    for dim, label in enumerate(("x (lateral)", "y (vertical)", "z (profundidad)")):
        ax = fig.add_subplot(3, 4, 6 + dim)
        for traj, sign, color in ((traj_a, sign_a, "C0"), (traj_b, sign_b, "C3")):
            if traj is None:
                continue
            mean, std = traj
            frames = np.arange(mean.shape[0])
            ax.plot(frames, mean[:, dim], color=color, lw=1.6, label=sign)
            ax.fill_between(
                frames,
                mean[:, dim] - std[:, dim],
                mean[:, dim] + std[:, dim],
                color=color,
                alpha=0.18,
            )
        ax.set_title(f"muñeca {label}", fontsize=8)
        ax.set_xlabel("frame", fontsize=7)
        ax.tick_params(labelsize=7)
        if dim == 0:
            ax.legend(fontsize=7)

    # Fila 3: separabilidad de todas las features. Asterisco = configuración de dedos.
    ax = fig.add_subplot(3, 1, 3)
    names = [r["feature"] for r in table]
    seps = [r["sep"] for r in table]
    colors = ["seagreen" if s >= 0.6 else "goldenrod" if s >= 0.3 else "indianred" for s in seps]
    ax.bar(np.arange(len(names)), seps, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(
        [f"{n} *" if n in HANDSHAPE_NAMES else n for n in names],
        rotation=55,
        ha="right",
        fontsize=6,
    )
    ax.axhline(0.6, color="0.4", ls="--", lw=0.8)
    ax.axhline(0.3, color="0.7", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("separabilidad", fontsize=8)
    ax.set_title(
        "verde >=0.60 separa bien | amarillo 0.30-0.60 parcial | rojo <0.30 no separa"
        "      (* = configuración de dedos, hoy no medida)",
        fontsize=8,
    )
    ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def mean_wrist_trajectory(data: dict, sign_idx: int):
    """Media y desvío de la muñeca activa (16, 3) para una seña."""
    from hand_face_features import HAND_WRIST, _as_blocks, pick_active_hand

    mask = data["raw_labels"] == sign_idx
    rows = data["raw"][mask]
    if not rows.size:
        return None
    n_frames = rows.shape[1] // 225
    trajs = []
    for flat in rows:
        seq = flat.reshape(n_frames, 225)
        points = np.full((n_frames, 3), np.nan, dtype=np.float32)
        for t in range(n_frames):
            _pose, left, right = _as_blocks(seq[t])
            hand = pick_active_hand(left, right)
            if hand is None:
                continue
            wrist = hand[HAND_WRIST]
            if not np.all(wrist == 0.0):
                points[t] = wrist
        trajs.append(points)
    stacked = np.stack(trajs)
    # Un frame puede no tener mano en ningún clip: nanmean avisa y devuelve nan.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
    return np.nan_to_num(mean), np.nan_to_num(std)


def plot_worst_signs(rows: list[dict], out_path: str, top: int = 30) -> None:
    worst = rows[:top]
    fig, ax = plt.subplots(figsize=(10, max(6, len(worst) * 0.3)))
    y = np.arange(len(worst))
    ax.barh(y, [r["acc"] for r in worst], color="indianred", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['sign']}  → {r['top_attractor']}" for r in worst], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("acc 1-NN leave-one-out (espacio crudo 225 dims)")
    ax.set_title(f"Señas más difíciles (peores {len(worst)}) y su atractor principal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_top_confusions(rows: list[dict], out_path: str, top: int = 30) -> None:
    worst = rows[:top]
    fig, ax = plt.subplots(figsize=(10, max(6, len(worst) * 0.3)))
    y = np.arange(len(worst))
    ax.barh(y, [r["n_confused"] for r in worst], color="steelblue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['a']} ↔ {r['b']}" for r in worst], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("clips cruzados (1-NN leave-one-out)")
    ax.set_title(f"Pares más confundidos (top {len(worst)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_heatmap(
    labels: np.ndarray, predicted: np.ndarray, signs: list[str], focus: list[str], out_path: str
) -> None:
    idxs = [signs.index(s) for s in focus if s in signs]
    if len(idxs) < 2:
        return
    matrix = np.zeros((len(idxs), len(idxs)), dtype=np.float32)
    for row, true_idx in enumerate(idxs):
        mask = labels == true_idx
        total = int(np.count_nonzero(mask))
        if not total:
            continue
        for col, pred_idx in enumerate(idxs):
            matrix[row, col] = np.count_nonzero(predicted[mask] == pred_idx) / total

    fig, ax = plt.subplots(figsize=(0.42 * len(idxs) + 4, 0.42 * len(idxs) + 3))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(idxs)))
    ax.set_yticks(np.arange(len(idxs)))
    ax.set_xticklabels([signs[i] for i in idxs], rotation=90, fontsize=7)
    ax.set_yticklabels([signs[i] for i in idxs], fontsize=7)
    ax.set_xlabel("vecino más cercano")
    ax.set_ylabel("seña real")
    ax.set_title("Matriz 1-NN (filas normalizadas). Fuera de la diagonal = riesgo")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------


def verdict(sep: float) -> str:
    if sep >= 0.8:
        return "SEPARABLE"
    if sep >= 0.6:
        return "separable con margen"
    if sep >= 0.3:
        return "parcial"
    return "NO SEPARABLE"


def describe_pair(data: dict, sign_a: str, sign_b: str, table: list[dict], cross: dict) -> str:
    best = table[0]
    base_best = next(r for r in table if r["feature"] in BASE_FEATURES)
    acc_12 = pair_combined_acc(data, sign_a, sign_b, BASE_FEATURES)
    acc_all = pair_combined_acc(data, sign_a, sign_b, ALL_FEATURES)
    lines = [
        f"### {sign_a} vs {sign_b}",
        "",
        f"- **En el espacio crudo (lo que ve el modelo): {100 - 100 * cross['rate']:.1f}% bien** "
        f"— solo {cross['n_confused']} de {cross['n_total']} clips cruzan.",
        f"- Combinando features: 12 actuales → {acc_12:.1%} | con dedos → {acc_all:.1%}.",
        f"- Mejor de las 12 actuales: **{base_best['feature']}** "
        f"({FEATURE_MEANING[base_best['feature']]}), sep {base_best['sep']:.2f} "
        f"→ {verdict(base_best['sep'])}.",
    ]
    if best["feature"] != base_best["feature"]:
        lines.append(
            f"- Con configuración de dedos: **{best['feature']}** "
            f"({FEATURE_MEANING[best['feature']]}), sep {best['sep']:.2f} → {verdict(best['sep'])}."
        )
    lines += [
        "",
        f"| feature | {sign_a} (media±sd) | {sign_b} (media±sd) | sep |",
        "|---|---|---|---|",
    ]
    for row in table[:6]:
        mark = " *" if row["feature"] in HANDSHAPE_NAMES else ""
        lines.append(
            f"| `{row['feature']}`{mark} | {row['mean_a']:.3f} ± {row['std_a']:.3f} "
            f"| {row['mean_b']:.3f} ± {row['std_b']:.3f} | {row['sep']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def cross_stats(data: dict, labels: np.ndarray, predicted: np.ndarray, sign_a: str, sign_b: str) -> dict:
    idx_a = data["signs"].index(sign_a)
    idx_b = data["signs"].index(sign_b)
    mask = (labels == idx_a) | (labels == idx_b)
    total = int(np.count_nonzero(mask))
    crossed = int(
        np.count_nonzero((labels == idx_a) & (predicted == idx_b))
        + np.count_nonzero((labels == idx_b) & (predicted == idx_a))
    )
    return {"n_confused": crossed, "n_total": total, "rate": crossed / total if total else 0.0}


def neighbours_of(sign: str, rows: list[dict], top: int = 5) -> list[dict]:
    hits = []
    for row in rows:
        if row["a"] == sign:
            hits.append({"other": row["b"], **row})
        elif row["b"] == sign:
            hits.append({"other": row["a"], **row})
    hits.sort(key=lambda r: r["n_confused"], reverse=True)
    return hits[:top]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confusiones entre señas sobre todo el dataset .npy.")
    parser.add_argument("--out", default="inspect_out/confusions", help="Carpeta de salida.")
    parser.add_argument("--max-clips", type=int, default=None, help="Tope de clips por seña (default: todos).")
    parser.add_argument("--top-pairs", type=int, default=30, help="Cuántos pares confusos reportar.")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help='Pares extra, formato "A,B". Se suman a los pedidos por defecto.',
    )
    parser.add_argument("--no-global", action="store_true", help="Salta el análisis global (solo pares).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_root = os.path.abspath(args.out)
    pairs_dir = os.path.join(out_root, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    print("Cargando dataset y calculando features...")
    data = load_dataset(args.max_clips)
    signs = data["signs"]
    print(
        f"\n{len(signs)} señas, {data['raw'].shape[0]} clips, "
        f"{data['raw'].shape[1]} dims crudas. Sin mano usable: {data['skipped']}."
    )

    print("1-NN leave-one-out en el espacio crudo (lo que ve el modelo)...")
    predicted = loo_1nn(data["raw"], data["raw_labels"])
    labels = data["raw_labels"]
    overall = float(np.mean(predicted == labels))
    print(f"Acc 1-NN global: {overall:.3f}")

    pair_rows = confusion_pairs(labels, predicted, signs)
    sign_rows = per_sign_accuracy(labels, predicted, signs)

    report = [
        "# Confusiones entre señas — dataset completo",
        "",
        f"- Clips: **{data['raw'].shape[0]}** en **{len(signs)}** señas "
        f"({data['raw'].shape[0] / len(signs):.0f} por seña en promedio).",
        f"- Acc 1-NN leave-one-out en el espacio crudo (225 dims × 16 frames): **{overall:.1%}**.",
        "- `sep` = |AUC − 0.5| × 2 sobre una feature. 1.0 separa perfecto, 0.0 no separa nada.",
        "- Features marcadas con `*` = configuración de dedos, **hoy no medida** por el inspector.",
        "",
        "> El 1-NN mide **cercanía geométrica**, no el modelo entrenado. Clips de la misma",
        "> sesión de grabación se parecen entre sí, así que el número absoluto es optimista;",
        "> lo informativo es **qué pares** aparecen cruzados.",
        "",
    ]

    if not args.no_global:
        report += ["## Señas más difíciles", "", "| seña | clips | acc 1-NN | atractor principal |", "|---|---|---|---|"]
        for row in sign_rows[:25]:
            report.append(
                f"| **{row['sign']}** | {row['n_clips']} | {row['acc']:.1%} | {row['top_attractor']} |"
            )
        report += [
            "",
            "## Pares más confundidos",
            "",
            "`sep 12` = mejor de las 12 features actuales. `sep +dedos` = mejor incluyendo",
            "el descriptor de configuración de dedos (hoy no medido).",
            "",
            "| par | clips cruzados | tasa | mejor de las 12 | sep 12 | mejor con dedos | sep +dedos | 1-NN solo con dedos |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in pair_rows[: args.top_pairs]:
            table = pair_feature_table(data, row["a"], row["b"])
            base_best = next(r for r in table if r["feature"] in BASE_FEATURES)
            acc_all = pair_combined_acc(data, row["a"], row["b"], ALL_FEATURES)
            report.append(
                f"| **{row['a']} ↔ {row['b']}** | {row['n_confused']} | {row['rate']:.1%} "
                f"| `{base_best['feature']}` | {base_best['sep']:.2f} "
                f"| `{table[0]['feature']}` | {table[0]['sep']:.2f} | {acc_all:.0%} |"
            )
        report.append("")

        # ¿Cuánto aporta medir dedos en los pares que hoy no se separan?
        gains = []
        for row in pair_rows[:60]:
            table = pair_feature_table(data, row["a"], row["b"])
            base_best = next(r for r in table if r["feature"] in BASE_FEATURES)
            gains.append(
                {
                    "pair": f"{row['a']} ↔ {row['b']}",
                    "base": base_best["sep"],
                    "full": table[0]["sep"],
                    "feature": table[0]["feature"],
                    "gain": table[0]["sep"] - base_best["sep"],
                }
            )
        gains.sort(key=lambda g: g["gain"], reverse=True)
        report += [
            "## ¿Sirve medir la configuración de dedos?",
            "",
            "Pares donde el descriptor de dedos gana más separabilidad frente a las 12 actuales.",
            "",
            "| par | sep con 12 | sep con dedos | gana | feature que lo resuelve |",
            "|---|---|---|---|---|",
        ]
        for g in gains[:20]:
            report.append(
                f"| {g['pair']} | {g['base']:.2f} | {g['full']:.2f} "
                f"| +{g['gain']:.2f} | `{g['feature']}` |"
            )
        report.append("")

        plot_worst_signs(sign_rows, os.path.join(out_root, "worst_signs.png"))
        plot_top_confusions(pair_rows, os.path.join(out_root, "top_confusions.png"), args.top_pairs)
        focus = [r["sign"] for r in sign_rows[:30]]
        plot_confusion_heatmap(labels, predicted, signs, focus, os.path.join(out_root, "confusion_heatmap.png"))

        with open(os.path.join(out_root, "per_sign.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sign", "n_clips", "acc", "top_attractor"])
            writer.writeheader()
            writer.writerows(sign_rows)
        with open(os.path.join(out_root, "pairs.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["a", "b", "n_confused", "rate"])
            writer.writeheader()
            writer.writerows(pair_rows)

    wanted_pairs = list(REQUESTED_PAIRS)
    for raw in args.pairs or []:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 2:
            wanted_pairs.append((parts[0], parts[1]))

    report += ["## Comparativas pedidas", ""]
    for raw_a, raw_b in wanted_pairs:
        sign_a = resolve_name(raw_a, signs)
        sign_b = resolve_name(raw_b, signs)
        if not sign_a or not sign_b:
            report.append(f"### {raw_a} vs {raw_b}\n\nNo encontré {'/'.join(x for x, y in ((raw_a, sign_a), (raw_b, sign_b)) if not y)} en el dataset.\n")
            continue
        table = pair_feature_table(data, sign_a, sign_b)
        cross = cross_stats(data, labels, predicted, sign_a, sign_b)
        report.append(describe_pair(data, sign_a, sign_b, table, cross))
        slug = f"{_fold(sign_a)}__vs__{_fold(sign_b)}.png"
        plot_pair(data, sign_a, sign_b, table, os.path.join(pairs_dir, slug))
        print(f"  par {sign_a} vs {sign_b}: sep={table[0]['sep']:.2f} ({table[0]['feature']})")

    report += ["## Señas sueltas: con quién se cruzan", ""]
    for raw in REQUESTED_SINGLES:
        sign = resolve_name(raw, signs)
        if not sign:
            report.append(f"### {raw}\n\nNo está en el dataset.\n")
            continue
        acc = next((r for r in sign_rows if r["sign"] == sign), None)
        hits = neighbours_of(sign, pair_rows)
        report.append(f"### {sign}")
        report.append("")
        if acc:
            report.append(f"- acc 1-NN: **{acc['acc']:.1%}** sobre {acc['n_clips']} clips.")
        if not hits:
            report.append("- No se cruza con nadie de forma apreciable.")
            report.append("")
            continue
        report.append("")
        report.append("| se cruza con | clips | mejor de las 12 | sep 12 | mejor con dedos | sep | veredicto |")
        report.append("|---|---|---|---|---|---|---|")
        for hit in hits:
            table = pair_feature_table(data, sign, hit["other"])
            base_best = next(r for r in table if r["feature"] in BASE_FEATURES)
            report.append(
                f"| {hit['other']} | {hit['n_confused']} | `{base_best['feature']}` "
                f"| {base_best['sep']:.2f} | `{table[0]['feature']}` "
                f"| {table[0]['sep']:.2f} | {verdict(table[0]['sep'])} |"
            )
            slug = f"{_fold(sign)}__vs__{_fold(hit['other'])}.png"
            path = os.path.join(pairs_dir, slug)
            if not os.path.exists(path):
                plot_pair(data, sign, hit["other"], table, path)
        report.append("")

    report_path = os.path.join(out_root, "REPORTE.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print(f"\nGuardado {report_path}")
    print(f"Gráficos en {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
