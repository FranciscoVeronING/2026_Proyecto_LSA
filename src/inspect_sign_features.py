"""
Inspección de geometría mano–rostro sobre .npy ya extraídos.

No reentrena ni cambia el input del modelo. Sirve para ver si I/T/ojo/quien
(y otras que fallan en cámara) ya se separan con offsets/distancias antes
de cablear feature engineering.

Uso (desde src/, entorno lsa_gpu):

    python inspect_sign_features.py --list
    python inspect_sign_features.py --dataset --no-show
    python inspect_sign_features.py --dataset --clips-per-sign 1 --no-show
    python inspect_sign_features.py --group face
    python inspect_sign_features.py --signs I T ojo

Modo --dataset: recorre todas las carpetas en dataset_landmarks_* (.npy),
genera inspect_out/dataset/by_sign/<seña>/ con gráficos por clase,
summary.csv global y comparativas en groups/ y overview/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from hand_face_features import (
    FEATURE_NAMES,
    MOTION_NAMES,
    clip_motion_stats,
    clip_summary_row,
    face_and_hand_xy,
    hand_face_features_sequence,
    peak_frame_index,
    visible_frame_mask,
)

SIGN_GROUPS = {
    "face": ["I", "T", "ojo", "quien", "papa"],
    "persist": ["I", "ellos", "papa", "repetir", "tener", "vos"],
    "same_form": ["O", "0"],
    "static_dyn": ["L", "lunes", "G", "años", "F", "donde", "V", "viernes"],
}
SIGN_GROUPS["homo"] = list(SIGN_GROUPS["static_dyn"])

DEFAULT_EVAL = os.path.join(
    "model",
    "2026_08_18_model_no_opt",
    "eval_94senias_20260828_222229_no_opt.csv",
)


def _npy_root() -> str:
    return os.path.abspath(cfg.DATASET_NPY_DIR)


def _video_root() -> str:
    return os.path.abspath(cfg.DATASET_VIDEOS_DIR)


def _match_dir(root: str, sign: str) -> str | None:
    """Las carpetas del dataset suelen ser 'Ojo' y las clases del modelo 'ojo'."""
    if not os.path.isdir(root):
        return None
    exact = os.path.join(root, sign)
    if os.path.isdir(exact):
        return exact
    target = sign.casefold()
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if os.path.isdir(folder) and name.casefold() == target:
            return folder
    return None


def _list_files(folder: str | None, suffix: str) -> list[str]:
    if not folder:
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(suffix) and "copia" not in name.lower()
    )


def _npy_paths(sign: str) -> list[str]:
    return _list_files(_match_dir(_npy_root(), sign), ".npy")


def _video_paths(sign: str) -> list[str]:
    return _list_files(_match_dir(_video_root(), sign), ".mp4")


def _signs_in_root(root: str, suffix: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    names = []
    for name in sorted(os.listdir(root), key=str.casefold):
        folder = os.path.join(root, name)
        if os.path.isdir(folder) and _list_files(folder, suffix):
            names.append(name)
    return names


def available_signs() -> list[str]:
    npy = _signs_in_root(_npy_root(), ".npy")
    if npy:
        return npy
    return _signs_in_root(_video_root(), ".mp4")


def _load_sequence(path: str) -> np.ndarray:
    seq = np.load(path)
    if seq.ndim != 2:
        raise ValueError(f"{path}: se esperaba (T, F), shape={seq.shape}")
    return seq.astype(np.float32)


def resolve_signs(args: argparse.Namespace) -> list[str]:
    chosen: list[str] = []
    if args.group:
        chosen.extend(SIGN_GROUPS[args.group])
    if args.signs:
        chosen.extend(args.signs)
    if args.from_eval:
        failed, _rows = failing_from_eval(args.from_eval, args.model)
        attractors = []
        if args.include_attractors:
            attractors = [row["top1"] for row in _rows]
        chosen.extend(failed)
        chosen.extend(attractors)
    if not chosen:
        chosen = list(SIGN_GROUPS["face"])

    seen = set()
    ordered = []
    for name in chosen:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def failing_from_eval(csv_path: str, model: str | None) -> tuple[list[str], list[dict]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No encontré el CSV: {csv_path}")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if model and raw.get("model") and raw["model"] != model:
                continue
            try:
                hit = int(float(raw["hit_top1"]))
            except (KeyError, ValueError):
                continue
            if hit != 0:
                continue
            rows.append(
                {
                    "expected": raw["expected_sign"],
                    "top1": raw["top1"],
                    "conf1": raw.get("conf1", ""),
                    "top2": raw.get("top2", ""),
                    "hit_top3": raw.get("hit_top3", ""),
                    "model": raw.get("model", ""),
                }
            )

    failed = []
    seen = set()
    for row in rows:
        name = row["expected"]
        if name not in seen:
            seen.add(name)
            failed.append(name)
    return failed, rows


def _select_paths(paths: list[str], sign: str, clip_indices, clip_files, max_clips: int | None):
    selected = paths
    if clip_files:
        wanted = {name.lower() for name in clip_files}
        selected = [p for p in paths if os.path.basename(p).lower() in wanted]
        if not selected:
            print(f"[!] Ningún archivo de --clip-files matcheó en '{sign}'.")
    elif clip_indices:
        selected = []
        for idx in clip_indices:
            if 0 <= idx < len(paths):
                selected.append(paths[idx])
            else:
                print(f"[!] {sign}: índice {idx} fuera de rango (0–{len(paths) - 1}).")
    if max_clips is not None and len(selected) > max_clips:
        selected = selected[:max_clips]
        print(f"[i] {sign}: usando {max_clips} clips (de {len(paths)}).")
    return selected


def resolve_source(sign: str, source: str) -> tuple[str, list[str]]:
    npy = _npy_paths(sign)
    videos = _video_paths(sign)
    if source == "npy":
        return "npy", npy
    if source == "video":
        return "video", videos
    if npy:
        return "npy", npy
    return "video", videos


def collect_clips(
    signs: list[str],
    clip_indices: list[int] | None,
    clip_files: list[str] | None,
    source: str,
    max_clips: int | None,
):
    records = []
    missing = []
    for sign in signs:
        kind, paths = resolve_source(sign, source)
        if not paths:
            missing.append(sign)
            continue
        selected = _select_paths(paths, sign, clip_indices, clip_files, max_clips)
        for path in selected:
            records.append(
                {
                    "sign": sign,
                    "path": path,
                    "name": os.path.basename(path),
                    "source": kind,
                }
            )
    if missing:
        npy_hint = ", ".join(_signs_in_root(_npy_root(), ".npy")[:12]) or "-"
        vid_hint = ", ".join(_signs_in_root(_video_root(), ".mp4")[:12]) or "-"
        raise FileNotFoundError(
            f"Sin .npy ni MP4 para: {', '.join(missing)}.\n"
            f"  npy:    {_npy_root()} ({npy_hint})\n"
            f"  videos: {_video_root()} ({vid_hint})"
        )
    return records


def _extract_from_video(path: str, target_frames: int, holistic) -> np.ndarray:
    from preprocessing import process_video_to_landmarks, _is_left_handed_video

    return process_video_to_landmarks(
        video_path=path,
        holistic_model=holistic,
        target_frames=target_frames,
        use_pose=cfg.USE_POSE,
        use_hands=cfg.USE_HANDS,
        use_face=False,
        left_handed=_is_left_handed_video(path),
    )


def analyze_records(records: list[dict], target_frames: int) -> list[dict]:
    need_holistic = any(rec["source"] == "video" for rec in records)
    holistic = None
    if need_holistic:
        import mediapipe as mp

        print(f"[i] Extrayendo landmarks de MP4 (T={target_frames}). Puede tardar.")
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    analyzed = []
    try:
        for rec in records:
            if rec["source"] == "npy":
                seq = _load_sequence(rec["path"])
            else:
                print(f"    {rec['sign']}/{rec['name']}")
                seq = _extract_from_video(rec["path"], target_frames, holistic)
            feats = hand_face_features_sequence(seq)
            summary = clip_summary_row(feats)
            peak = peak_frame_index(feats)
            visible = visible_frame_mask(feats)
            analyzed.append(
                {
                    **rec,
                    "seq": seq,
                    "feats": feats,
                    "summary": summary,
                    "motion": clip_motion_stats(seq),
                    "peak": peak,
                    "n_visible": int(np.count_nonzero(visible)),
                    "n_frames": int(feats.shape[0]),
                }
            )
    finally:
        if holistic is not None:
            holistic.close()
    return analyzed


def print_summary_table(analyzed: list[dict]) -> None:
    by_sign = defaultdict(list)
    for item in analyzed:
        by_sign[item["sign"]].append(item)

    header = (
        f"{'seña':<12} {'clips':>5} {'mano%':>6} "
        + " ".join(f"{name:>16}" for name in FEATURE_NAMES)
    )
    print()
    print(header)
    print("-" * len(header))
    for sign in by_sign:
        items = by_sign[sign]
        rows = [it["summary"] for it in items if it["summary"] is not None]
        n = len(items)
        n_ok = len(rows)
        hand_pct = 100.0 * n_ok / n if n else 0.0
        if rows:
            stacked = np.stack(rows, axis=0)
            means = stacked.mean(axis=0)
            cells = [f"{v:16.3f}" for v in means]
        else:
            cells = [f"{'-':>16}" for _ in FEATURE_NAMES]
        print(f"{sign:<12} {n:5d} {hand_pct:5.0f}% " + " ".join(cells))
    print()
    print("Valores = media del frame más cerca del ojo (dist_index_eye).")
    print("Y crece hacia abajo: dy_index_mouth < 0 => indice mas arriba que la boca.")
    print()
    motion_header = (
        f"{'seña':<12} {'clips':>5} {'ok%':>5} "
        + " ".join(f"{name:>14}" for name in MOTION_NAMES)
    )
    print(motion_header)
    print("-" * len(motion_header))
    for sign in by_sign:
        items = by_sign[sign]
        rows = [it["motion"] for it in items if it.get("motion") is not None]
        n = len(items)
        n_ok = len(rows)
        ok_pct = 100.0 * n_ok / n if n else 0.0
        if rows:
            stacked = np.stack(rows, axis=0)
            means = stacked.mean(axis=0)
            cells = [f"{v:14.3f}" for v in means]
        else:
            cells = [f"{'-':>14}" for _ in MOTION_NAMES]
        print(f"{sign:<12} {n:5d} {ok_pct:4.0f}% " + " ".join(cells))
    print()
    print("Movimiento = std/rango de la muñeca activa a lo largo del clip.")
    print("L/lunes ~ eje X; G/años ~ eje Y; F/donde ~ Z pecho; V/viernes ~ Z menton (2 golpes).")


def _colors_for(signs: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {sign: cmap(i % 10) for i, sign in enumerate(signs)}


def plot_histograms(analyzed: list[dict], signs: list[str], out_path: str | None) -> None:
    colors = _colors_for(signs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Distribución por seña (1 valor por video, frame más cerca del ojo)")

    for ax, name in zip(axes.ravel(), FEATURE_NAMES):
        for sign in signs:
            values = [
                float(item["summary"][FEATURE_NAMES.index(name)])
                for item in analyzed
                if item["sign"] == sign and item["summary"] is not None
            ]
            if not values:
                continue
            ax.hist(
                values,
                bins=min(12, max(5, len(values) // 3 or 5)),
                alpha=0.45,
                label=sign,
                color=colors[sign],
                density=True,
            )
        ax.set_title(name)
        ax.set_xlabel("unidades de hombro")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"Guardado {out_path}")
        plt.close(fig)


def plot_motion_histograms(analyzed: list[dict], signs: list[str], out_path: str | None) -> None:
    colors = _colors_for(signs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Movimiento de la muñeca por seña (1 valor por video)")

    for ax, name in zip(axes.ravel(), MOTION_NAMES):
        for sign in signs:
            values = [
                float(item["motion"][MOTION_NAMES.index(name)])
                for item in analyzed
                if item["sign"] == sign and item.get("motion") is not None
            ]
            if not values:
                continue
            ax.hist(
                values,
                bins=min(12, max(5, len(values) // 3 or 5)),
                alpha=0.45,
                label=sign,
                color=colors[sign],
                density=True,
            )
        ax.set_title(name)
        ax.set_xlabel("unidades de hombro")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"Guardado {out_path}")
        plt.close(fig)


def plot_scatter(analyzed: list[dict], signs: list[str], out_path: str | None) -> None:
    colors = _colors_for(signs)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for sign in signs:
        xs, ys = [], []
        for item in analyzed:
            if item["sign"] != sign or item["summary"] is None:
                continue
            xs.append(float(item["summary"][FEATURE_NAMES.index("dy_index_mouth")]))
            ys.append(float(item["summary"][FEATURE_NAMES.index("dist_index_eye")]))
        if xs:
            ax.scatter(xs, ys, label=sign, color=colors[sign], alpha=0.75, s=36)
    ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("dy_index_mouth  (<0 = índice arriba de la boca)")
    ax.set_ylabel("dist_index_eye")
    ax.set_title("¿Se separan las nubes?")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"Guardado {out_path}")
        plt.close(fig)


def plot_timeseries(
    analyzed: list[dict],
    signs: list[str],
    feature_name: str,
    max_lines: int,
    out_path: str | None,
) -> None:
    colors = _colors_for(signs)
    feat_idx = FEATURE_NAMES.index(feature_name)
    fig, ax = plt.subplots(figsize=(11, 5))
    drawn = 0
    for item in analyzed:
        if drawn >= max_lines:
            break
        feats = item["feats"]
        mask = visible_frame_mask(feats)
        if not np.any(mask):
            continue
        frames = np.arange(feats.shape[0])
        y = feats[:, feat_idx].astype(float)
        y[~mask] = np.nan
        ax.plot(
            frames,
            y,
            color=colors[item["sign"]],
            alpha=0.55,
            lw=1.2,
            label=item["sign"] if item["sign"] not in ax.get_legend_handles_labels()[1] else None,
        )
        drawn += 1
    ax.set_xlabel("frame (después del trim/subsampleo)")
    ax.set_ylabel(feature_name)
    ax.set_title(f"Serie temporal — {feature_name} (hasta {max_lines} clips)")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"Guardado {out_path}")
        plt.close(fig)


def plot_snapshots(
    analyzed: list[dict],
    signs: list[str],
    per_sign: int,
    out_path: str | None,
) -> None:
    colors = _colors_for(signs)
    chosen = []
    counts = defaultdict(int)
    for item in analyzed:
        if item["peak"] is None:
            continue
        if counts[item["sign"]] >= per_sign:
            continue
        counts[item["sign"]] += 1
        chosen.append(item)
    if not chosen:
        print("[!] No hay frames con mano para el snapshot 2D.")
        return

    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    fig.suptitle("Snapshot 2D en el frame más cerca del ojo (Y invertido como en imagen)")

    for ax, item in zip(axes_list, chosen):
        peak = item["peak"]
        pts = face_and_hand_xy(item["seq"][peak])
        color = colors[item["sign"]]
        ax.set_title(f"{item['sign']}\n{item['name']}", fontsize=8)
        if not pts:
            ax.text(0.5, 0.5, "sin pose", ha="center", va="center")
            ax.axis("off")
            continue
        for key in ("shoulder_l", "shoulder_r", "nose", "left_eye", "right_eye", "mouth_l", "mouth_r"):
            x, y = pts[key]
            ax.scatter([x], [y], s=18, c="0.35")
        ax.plot(
            [pts["mouth_l"][0], pts["mouth_r"][0]],
            [pts["mouth_l"][1], pts["mouth_r"][1]],
            color="0.35",
            lw=1,
        )
        if "index" in pts:
            ax.scatter([pts["index"][0]], [pts["index"][1]], s=70, marker="*", c=[color])
            ax.plot(
                [pts["index"][0], pts["eye"][0]],
                [pts["index"][1], pts["eye"][1]],
                color=color,
                lw=1.2,
            )
            ax.plot(
                [pts["index"][0], pts["mouth"][0]],
                [pts["index"][1], pts["mouth"][1]],
                color=color,
                lw=1.0,
                ls="--",
            )
        if "wrist" in pts:
            ax.scatter([pts["wrist"][0]], [pts["wrist"][1]], s=22, c=[color])
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_list[len(chosen) :]:
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"Guardado {out_path}")
        plt.close(fig)


def sanitize_sign_dir(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return safe.replace(" ", "_") or "sign"


def sign_matches(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def resolve_group_signs(group_key: str, available: list[str]) -> list[str]:
    wanted = SIGN_GROUPS.get(group_key, [])
    out = []
    for w in wanted:
        for folder in available:
            if sign_matches(w, folder) and folder not in out:
                out.append(folder)
                break
    return out


def collect_dataset_clips(clips_per_sign: int, source: str) -> list[dict]:
    signs = _signs_in_root(_npy_root(), ".npy")
    if not signs and source != "video":
        signs = _signs_in_root(_video_root(), ".mp4")
    if not signs:
        raise FileNotFoundError(
            f"No hay carpetas con .npy en {_npy_root()}. Corré preprocessing.py primero."
        )
    records = []
    for sign in signs:
        kind, paths = resolve_source(sign, source)
        if not paths:
            continue
        n = min(clips_per_sign, len(paths))
        if n == 1:
            indices = [0]
        else:
            indices = [int(round(i * (len(paths) - 1) / (n - 1))) for i in range(n)]
        for idx in indices:
            records.append(
                {
                    "sign": sign,
                    "path": paths[idx],
                    "name": os.path.basename(paths[idx]),
                    "source": kind,
                }
            )
    return records


def _wrist_trajectory(sequence: np.ndarray) -> np.ndarray | None:
    from hand_face_features import HAND_WRIST, _as_blocks, pick_active_hand

    points = []
    for t in range(sequence.shape[0]):
        _pose, left_hand, right_hand = _as_blocks(sequence[t])
        hand = pick_active_hand(left_hand, right_hand)
        if hand is None:
            points.append([np.nan, np.nan, np.nan])
            continue
        wrist = hand[HAND_WRIST]
        if np.all(wrist == 0.0):
            points.append([np.nan, np.nan, np.nan])
        else:
            points.append(wrist.tolist())
    arr = np.asarray(points, dtype=np.float32)
    if np.all(np.isnan(arr)):
        return None
    return arr


def plot_sign_hand_face_series(items: list[dict], out_path: str, title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=10)
    for item in items:
        feats = item["feats"]
        mask = visible_frame_mask(feats)
        frames = np.arange(feats.shape[0])
        for ax, idx in zip(axes.ravel(), range(len(FEATURE_NAMES))):
            y = feats[:, idx].astype(float)
            y[~mask] = np.nan
            ax.plot(frames, y, alpha=0.8, lw=1.2, label=item["name"][:20])
            ax.set_title(FEATURE_NAMES[idx], fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("frame")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sign_wrist_series(items: list[dict], out_path: str, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=10)
    labels = ("x (lateral)", "y (vertical)", "z (profundidad)")
    for item in items:
        traj = _wrist_trajectory(item["seq"])
        if traj is None:
            continue
        frames = np.arange(traj.shape[0])
        for ax, dim, lab in zip(axes, range(3), labels):
            ax.plot(frames, traj[:, dim], alpha=0.8, lw=1.2, label=item["name"][:20])
            ax.set_ylabel(lab, fontsize=8)
    axes[-1].set_xlabel("frame")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sign_snapshots(items: list[dict], out_path: str, title: str) -> None:
    chosen = [it for it in items if it.get("peak") is not None][:4]
    if not chosen:
        return
    cols = len(chosen)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
    axes_list = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=10)
    for ax, item in zip(axes_list, chosen):
        pts = face_and_hand_xy(item["seq"][item["peak"]])
        ax.set_title(item["name"][:24], fontsize=7)
        if not pts:
            ax.text(0.5, 0.5, "sin pose", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue
        for key in ("shoulder_l", "shoulder_r", "nose", "left_eye", "right_eye", "mouth_l", "mouth_r"):
            x, y = pts[key]
            ax.scatter([x], [y], s=14, c="0.35")
        ax.plot(
            [pts["mouth_l"][0], pts["mouth_r"][0]],
            [pts["mouth_l"][1], pts["mouth_r"][1]],
            color="0.35",
            lw=1,
        )
        if "index" in pts:
            ax.scatter([pts["index"][0]], [pts["index"][1]], s=60, marker="*", c="C0")
            ax.plot(
                [pts["index"][0], pts["eye"][0]],
                [pts["index"][1], pts["eye"][1]],
                color="C0",
                lw=1,
            )
        if "wrist" in pts:
            ax.scatter([pts["wrist"][0]], [pts["wrist"][1]], s=18, c="C1")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_item_metrics_json(items: list[dict], path: str) -> None:
    payload = []
    for item in items:
        row = {
            "clip": item["name"],
            "n_frames": item["n_frames"],
            "n_visible_frames": item["n_visible"],
            "peak_frame": item["peak"],
        }
        if item["summary"] is not None:
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = float(item["summary"][i])
        if item.get("motion") is not None:
            for i, name in enumerate(MOTION_NAMES):
                row[name] = float(item["motion"][i])
        payload.append(row)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_summary_csv(analyzed: list[dict], path: str) -> None:
    by_sign = defaultdict(list)
    for item in analyzed:
        by_sign[item["sign"]].append(item)

    fieldnames = (
        ["sign", "n_clips", "hand_pct"]
        + [f"mean_{n}" for n in FEATURE_NAMES]
        + [f"mean_{n}" for n in MOTION_NAMES]
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sign in sorted(by_sign.keys(), key=str.casefold):
            items = by_sign[sign]
            summaries = [it["summary"] for it in items if it["summary"] is not None]
            motions = [it["motion"] for it in items if it.get("motion") is not None]
            row = {
                "sign": sign,
                "n_clips": len(items),
                "hand_pct": round(100.0 * len(summaries) / len(items), 1) if items else 0,
            }
            if summaries:
                mean_s = np.stack(summaries, axis=0).mean(axis=0)
                for i, name in enumerate(FEATURE_NAMES):
                    row[f"mean_{name}"] = round(float(mean_s[i]), 4)
            if motions:
                mean_m = np.stack(motions, axis=0).mean(axis=0)
                for i, name in enumerate(MOTION_NAMES):
                    row[f"mean_{name}"] = round(float(mean_m[i]), 4)
            writer.writerow(row)
    print(f"Guardado {path}")


def plot_overview_motion_bars(analyzed: list[dict], out_dir: str) -> None:
    by_sign = defaultdict(list)
    for item in analyzed:
        if item.get("motion") is not None:
            by_sign[item["sign"]].append(item["motion"])

    for motion_idx, motion_name in enumerate(MOTION_NAMES):
        signs = []
        values = []
        for sign in sorted(by_sign.keys(), key=str.casefold):
            arr = np.stack(by_sign[sign], axis=0)
            signs.append(sign)
            values.append(float(arr[:, motion_idx].mean()))

        if not signs:
            continue
        order = np.argsort(values)[::-1]
        signs = [signs[i] for i in order]
        values = [values[i] for i in order]

        height = max(6, len(signs) * 0.18)
        fig, ax = plt.subplots(figsize=(10, height))
        y_pos = np.arange(len(signs))
        ax.barh(y_pos, values, color="steelblue", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(signs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("media por seña (unidades de hombro)")
        ax.set_title(f"Overview — {motion_name}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"overview_{motion_name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Guardado {path}")


def run_dataset_inspection(args: argparse.Namespace) -> int:
    clips_per = args.clips_per_sign
    source = "npy" if args.source == "auto" else args.source
    try:
        records = collect_dataset_clips(clips_per, source)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    signs = sorted({r["sign"] for r in records}, key=str.casefold)
    root = os.path.abspath(args.out)
    by_sign_root = os.path.join(root, "by_sign")
    groups_root = os.path.join(root, "groups")
    overview_root = os.path.join(root, "overview")
    os.makedirs(by_sign_root, exist_ok=True)
    os.makedirs(groups_root, exist_ok=True)
    os.makedirs(overview_root, exist_ok=True)

    print(f"Dataset: {len(signs)} señas, {len(records)} clips ({clips_per} por seña)")
    print(f"Salida:  {root}")
    print(f"Fuente:  {_npy_root()}")

    analyzed = analyze_records(records, args.frames)
    by_sign = defaultdict(list)
    for item in analyzed:
        by_sign[item["sign"]].append(item)

    for sign, items in sorted(by_sign.items(), key=lambda x: x[0].casefold()):
        sign_dir = os.path.join(by_sign_root, sanitize_sign_dir(sign))
        os.makedirs(sign_dir, exist_ok=True)
        write_item_metrics_json(items, os.path.join(sign_dir, "metrics.json"))
        plot_sign_snapshots(items, os.path.join(sign_dir, "snapshot.png"), sign)
        plot_sign_hand_face_series(
            items, os.path.join(sign_dir, "series_hand_face.png"), f"{sign} — mano/rostro"
        )
        plot_sign_wrist_series(
            items, os.path.join(sign_dir, "series_wrist.png"), f"{sign} — muñeca xyz"
        )

    write_summary_csv(analyzed, os.path.join(root, "summary.csv"))
    print_summary_table(analyzed)
    plot_overview_motion_bars(analyzed, overview_root)

    available = list(by_sign.keys())
    for group_key in ("face", "static_dyn", "persist", "same_form"):
        group_signs = resolve_group_signs(group_key, available)
        if len(group_signs) < 2:
            continue
        group_items = [it for it in analyzed if it["sign"] in group_signs]
        group_dir = os.path.join(groups_root, group_key)
        os.makedirs(group_dir, exist_ok=True)
        plot_histograms(group_items, group_signs, os.path.join(group_dir, "histograms_hand_face.png"))
        plot_motion_histograms(group_items, group_signs, os.path.join(group_dir, "histograms_motion.png"))
        plot_scatter(group_items, group_signs, os.path.join(group_dir, "scatter_mouth_eye.png"))
        plot_timeseries(
            group_items,
            group_signs,
            args.series,
            args.max_series,
            os.path.join(group_dir, f"series_{args.series}.png"),
        )
        plt.close("all")

    readme = os.path.join(root, "README.txt")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write(
            f"Inspeccion dataset — {datetime.now().isoformat(timespec='seconds')}\n"
            f"Clips por seña: {clips_per}\n"
            f"Señas: {len(signs)}\n\n"
            "by_sign/<seña>/\n"
            "  metrics.json       valores por clip\n"
            "  snapshot.png       pose 2D (frame cerca del ojo)\n"
            "  series_hand_face.png\n"
            "  series_wrist.png\n\n"
            "summary.csv          tabla global\n"
            "overview/            barras de movimiento por seña\n"
            "groups/              comparativas (face, static_dyn, …)\n"
        )
    print(f"Guardado {readme}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")
    return 0


def print_eval_errors(rows: list[dict]) -> None:
    if not rows:
        print("En ese CSV no hay fallos top-1 (con el filtro de --model).")
        return
    print("\nFallos top-1 del eval:")
    print(f"{'esperada':<12} {'-> top-1':<12} {'conf':>6}  top-3?  modelo")
    for row in rows:
        print(
            f"{row['expected']:<12} {row['top1']:<12} {row['conf1']:>6}  "
            f"{row['hit_top3']:>5}  {row['model']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Histogramas y geometría mano–rostro de señas (sobre .npy)."
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Inspecciona todas las señas del dataset .npy (carpeta por seña + resumen global).",
    )
    parser.add_argument(
        "--clips-per-sign",
        type=int,
        default=1,
        help="Clips por seña en modo --dataset (default 1; usar 5–10 para ver variabilidad).",
    )
    parser.add_argument("--list", action="store_true", help="Lista clases con .npy y sale.")
    parser.add_argument(
        "--list-clips",
        metavar="SEÑA",
        help="Lista los .npy de una seña (con índice) y sale.",
    )
    parser.add_argument("--signs", nargs="+", help="Señas a comparar, ej. I T ojo")
    parser.add_argument(
        "--group",
        choices=sorted(SIGN_GROUPS),
        help="Atajo: face; persist; same_form=O/0; static_dyn=L/lunes G/años F/donde V/viernes.",
    )
    parser.add_argument(
        "--from-eval",
        nargs="?",
        const=DEFAULT_EVAL,
        help="CSV de camera.py --eval. Sin ruta usa el baseline 28/08. Agrega las señas que fallaron.",
    )
    parser.add_argument(
        "--model",
        help="En evals apareadas, quedarse con un checkpoint (ej. 2026_08_18_model_no_opt).",
    )
    parser.add_argument(
        "--include-attractors",
        action="store_true",
        default=True,
        help="Con --from-eval, también grafica el top-1 erróneo (quien, documento, …).",
    )
    parser.add_argument(
        "--no-attractors",
        action="store_false",
        dest="include_attractors",
        help="No agregar las clases atractor del eval.",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        type=int,
        help="Índices 0-based sobre la lista ordenada de cada seña (ver --list-clips).",
    )
    parser.add_argument(
        "--clip-files",
        nargs="+",
        help="Nombres de archivo .npy concretos.",
    )
    parser.add_argument(
        "--series",
        default="dy_index_mouth",
        choices=FEATURE_NAMES,
        help="Feature para el gráfico temporal.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=18,
        help="Tope de líneas en el temporal (default 18).",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=2,
        help="Snapshots 2D por seña (default 2).",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "npy", "video"),
        default="auto",
        help="auto = .npy si existen, si no MP4 (default).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Tope de videos por seña. Recomendado si leés MP4 (ej. 10).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Frames al extraer desde MP4 (default 16, el baseline de cámara).",
    )
    parser.add_argument(
        "--out",
        default="inspect_out/dataset",
        help="Carpeta de salida (default: inspect_out/dataset con --dataset, si no inspect_out).",
    )
    parser.add_argument("--no-show", action="store_true", help="Solo guardar, no abrir ventanas.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.dataset and args.out == "inspect_out/dataset":
        args.out = "inspect_out"

    if args.dataset:
        return run_dataset_inspection(args)

    if args.list:
        npy_signs = _signs_in_root(_npy_root(), ".npy")
        vid_signs = _signs_in_root(_video_root(), ".mp4")
        print(f"npy:    {_npy_root()}")
        if npy_signs:
            for sign in npy_signs:
                print(f"  {sign:<16} {len(_npy_paths(sign)):3d} .npy")
        else:
            print("  (vacio - se pueden leer MP4 con --source auto)")
        print(f"videos: {_video_root()}")
        if vid_signs:
            for sign in vid_signs:
                print(f"  {sign:<16} {len(_video_paths(sign)):3d} .mp4")
        else:
            print("  (vacio)")
        if not npy_signs and not vid_signs:
            return 1
        return 0

    if args.list_clips:
        npy = _npy_paths(args.list_clips)
        videos = _video_paths(args.list_clips)
        if not npy and not videos:
            print(f"Sin .npy ni MP4 para '{args.list_clips}'")
            return 1
        if npy:
            print(f"{args.list_clips} (.npy): {len(npy)}")
            for idx, path in enumerate(npy):
                print(f"  {idx:3d}  {os.path.basename(path)}")
        if videos:
            print(f"{args.list_clips} (mp4): {len(videos)}")
            for idx, path in enumerate(videos):
                print(f"  {idx:3d}  {os.path.basename(path)}")
        return 0

    eval_rows = []
    if args.from_eval:
        try:
            _failed, eval_rows = failing_from_eval(args.from_eval, args.model)
        except FileNotFoundError as exc:
            print(exc)
            return 1
        print_eval_errors(eval_rows)

    try:
        signs = resolve_signs(args)
        max_clips = args.max_clips
        if (
            max_clips is None
            and not args.clips
            and not args.clip_files
            and any(resolve_source(sign, args.source)[0] == "video" for sign in signs)
        ):
            max_clips = 12
            print("[i] Sin .npy: tope automático de 12 clips/seña (cambialo con --max-clips).")
        records = collect_clips(
            signs,
            args.clips,
            args.clip_files,
            args.source,
            max_clips,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if not records:
        print("No quedó ningún clip para graficar.")
        return 1

    sources = sorted({rec["source"] for rec in records})
    print(f"Clases: {', '.join(signs)}")
    print(f"Clips:  {len(records)}  fuente={','.join(sources)}")
    if "video" in sources and args.max_clips is None and not args.clips and not args.clip_files:
        print("[i] Sin .npy: cada MP4 pasa por MediaPipe. Probá --max-clips 10 la primera vez.")
    analyzed = analyze_records(records, args.frames)
    print_summary_table(analyzed)

    os.makedirs(args.out, exist_ok=True)
    plot_histograms(analyzed, signs, os.path.join(args.out, "histograms.png"))
    plot_scatter(analyzed, signs, os.path.join(args.out, "scatter_mouth_eye.png"))
    plot_motion_histograms(analyzed, signs, os.path.join(args.out, "histograms_motion.png"))
    plot_timeseries(
        analyzed,
        signs,
        args.series,
        args.max_series,
        os.path.join(args.out, f"series_{args.series}.png"),
    )
    plot_snapshots(
        analyzed,
        signs,
        args.snapshots,
        os.path.join(args.out, "snapshots_2d.png"),
    )

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
