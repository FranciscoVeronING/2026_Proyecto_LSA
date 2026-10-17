"""Marca inicio/fin utiles de cada seña con la logica de VideoDivider.

VideoDivider parte un video largo en rafagas de movimiento de manos. En el
diccionario cada mp4 ya es UNA seña (o un compuesto pegado). No hay que
partir: hay que tomar desde la primera rafaga hasta la ultima, y conservar
las pausas internas (sosten de una letra, seña de dos movimientos).

Trabaja sobre los `.pose` ya extraidos (Holistic a ~30 fps), no vuelve a
correr HandLandmarker. El recorte se aplica al unir señas, no pisa videos.

  py -3.11 extraction/trim_sign_windows.py
  py -3.11 extraction/trim_sign_windows.py --only HOLA,MAMÁ,A,Agua
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from render_pose_pair import (  # noqa: E402
    HAND_POINTS,
    LEFT_HAND_START,
    RIGHT_HAND_START,
    WRIST_LEFT_IDX,
    WRIST_RIGHT_IDX,
    attach_empty_hands_to_wrists,
    fill_missing_landmarks,
    load_pose,
)

POSE_DIR = ROOT / "out" / "poses"
REPORT = ROOT / "out" / "sign_windows.json"

PAUSE_SECONDS = 0.20
MIN_SIGNAL_SECONDS = 0.20
PRE_ROLL = 0.15
POST_ROLL = 0.20
MIN_KEEP_SECONDS = 0.40
WRIST_MIN_PX = 18.0
WRIST_PEAK_FRAC = 0.15
HAND_MIN_CHANGE = 0.008
MISSING_CONF = 0.15
MISSING_XY = 4.0


def hand_scale(points: np.ndarray) -> float:
    width, height = np.ptp(points[:, :2], axis=0)
    return max(float(np.hypot(width, height)), 1e-3)


def hand_change(old: np.ndarray, new: np.ndarray) -> float:
    displacement = np.linalg.norm(new[:, :2] - old[:, :2], axis=1)
    return float(np.mean(displacement) / max(hand_scale(old), hand_scale(new)))


def hand_ok(points: np.ndarray, conf: np.ndarray) -> bool:
    if points.shape[0] < HAND_POINTS:
        return False
    xy = np.linalg.norm(points[:, :2], axis=1)
    return bool(np.mean(xy) >= MISSING_XY and np.mean(conf) >= MISSING_CONF)


def automatic_threshold(scores: np.ndarray, minimum: float, sigma: float) -> float:
    values = scores[scores > 0]
    if not len(values):
        return float(minimum)
    baseline = float(np.percentile(values, 25))
    spread = 1.4826 * float(np.median(np.abs(values - baseline)))
    return max(float(minimum), baseline + sigma * spread)


def rolling(values: np.ndarray, window: int, reducer) -> np.ndarray:
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = int(window) | 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray([reducer(padded[i : i + window]) for i in range(len(values))])


def detect_bursts(scores: np.ndarray, fps: float, start_thr: float, stop_thr: float) -> list[tuple[int, int]]:
    """Rafagas [start, end) como VideoDivider.detect, sin pre/post-roll."""
    if len(scores) < 2:
        return []
    starts = rolling(scores, 3, np.max)
    stops = rolling(scores, max(3, round(fps / 5)), np.median)
    stop_thr = max(float(start_thr), float(stop_thr))
    pause_n = max(1, math.ceil(PAUSE_SECONDS * fps))
    min_n = max(1, math.ceil(MIN_SIGNAL_SECONDS * fps))
    runs: list[tuple[int, int]] = []
    active = None
    below = 0
    for i, (start_value, stop_value) in enumerate(zip(starts, stops)):
        if active is None:
            if start_value >= start_thr:
                active = i
            continue
        if stop_value < stop_thr:
            below += 1
            if below >= pause_n:
                end = i - below + 1
                if end - active >= min_n:
                    runs.append((active, end))
                active, below = None, 0
        else:
            below = 0
    if active is not None and len(stops) - active >= min_n:
        runs.append((active, len(stops)))
    return runs


def motion_scores(data: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cambio de manos (VideoDivider) y distancia de muñecas al reposo."""
    frames = data.shape[0]
    scores = np.zeros(frames, dtype=np.float32)
    for t in range(1, frames):
        changes = []
        for start in (LEFT_HAND_START, RIGHT_HAND_START):
            sl = slice(start, start + HAND_POINTS)
            old_p, new_p = data[t - 1, 0, sl, :], data[t, 0, sl, :]
            old_c, new_c = conf[t - 1, 0, sl], conf[t, 0, sl]
            if hand_ok(old_p, old_c) and hand_ok(new_p, new_c):
                changes.append(hand_change(old_p, new_p))
        scores[t] = max(changes) if changes else 0.0

    wrists = data[:, 0, [WRIST_LEFT_IDX, WRIST_RIGHT_IDX], :2]
    edge = max(5, frames // 8)
    first, last = wrists[:edge], wrists[-edge:]
    rest = np.median(first if first[:, :, 1].mean() >= last[:, :, 1].mean() else last, axis=0)
    displacement = np.mean(np.linalg.norm(wrists - rest, axis=-1), axis=1).astype(np.float32)
    return scores, displacement


def useful_window(data: np.ndarray, conf: np.ndarray, fps: float) -> dict:
    frames = int(data.shape[0])
    scores, displacement = motion_scores(data, conf)
    peak = float(displacement.max()) if frames else 0.0
    wrist_thr = max(WRIST_MIN_PX, WRIST_PEAK_FRAC * peak)
    hand_thr = automatic_threshold(scores, HAND_MIN_CHANGE, 1.0)
    stop_thr = automatic_threshold(scores, HAND_MIN_CHANGE, 1.5)
    bursts = detect_bursts(scores, fps, hand_thr, max(hand_thr, stop_thr))
    wrist_on = displacement >= wrist_thr

    pre_n = max(0, round(PRE_ROLL * fps))
    post_n = max(0, round(POST_ROLL * fps))
    min_keep = max(8, round(MIN_KEEP_SECONDS * fps))

    core_start: int | None = None
    core_end: int | None = None
    if wrist_on.any():
        idx = np.where(wrist_on)[0]
        core_start, core_end = int(idx[0]), int(idx[-1]) + 1
    gap_n = max(1, round(0.35 * fps))
    for burst_start, burst_end in bursts:
        if core_start is None:
            core_start, core_end = burst_start, burst_end
            continue
        if burst_end + gap_n < core_start or burst_start - gap_n > core_end:
            continue
        core_start = min(core_start, burst_start)
        core_end = max(core_end, burst_end)

    if core_start is None:
        return {
            "start": 0,
            "end": frames,
            "kept": frames,
            "kept_frac": 1.0,
            "bursts": 0,
            "fallback": "sin_movimiento",
            "hand_thr": round(hand_thr, 5),
            "wrist_thr": round(wrist_thr, 2),
        }

    start = max(0, core_start - pre_n)
    end = min(frames, core_end + post_n)
    if end - start < min_keep:
        extra = min_keep - (end - start)
        start = max(0, start - extra // 2)
        end = min(frames, start + min_keep)
        if end - start < min_keep:
            start = max(0, frames - min_keep)
            end = frames

    return {
        "start": start,
        "end": end,
        "kept": end - start,
        "kept_frac": round((end - start) / frames, 4) if frames else 1.0,
        "bursts": len(bursts),
        "fallback": None,
        "hand_thr": round(hand_thr, 5),
        "wrist_thr": round(wrist_thr, 2),
        "cut_head": start,
        "cut_tail": frames - end,
    }


def window_for_pose(path: Path) -> dict:
    pose = load_pose(path)
    pose, _ = fill_missing_landmarks(pose, MISSING_CONF, MISSING_XY)
    pose = attach_empty_hands_to_wrists(pose, MISSING_CONF, MISSING_XY)
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    fps = float(pose.body.fps or 30.0)
    info = useful_window(data, conf, fps)
    info.update(
        {
            "stem": path.stem,
            "frames": int(data.shape[0]),
            "fps": round(fps, 3),
            "start_sec": round(info["start"] / fps, 3),
            "end_sec": round(info["end"] / fps, 3),
        }
    )
    return info


def load_sign_windows(path: Path | None = None) -> dict[str, dict]:
    report = path or REPORT
    if not report.exists():
        return {}
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {item["stem"]: item for item in payload.get("items", [])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="Stems separados por coma.")
    parser.add_argument("--output", type=Path, default=REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poses = sorted(p for p in POSE_DIR.glob("*.pose") if not p.stem.endswith("_main"))
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        poses = [path for path in poses if path.stem in wanted]
        missing = wanted - {path.stem for path in poses}
        if missing:
            print(f"no estan: {sorted(missing)}")
    if not poses:
        raise SystemExit(f"No hay .pose en {POSE_DIR}")

    items = []
    for index, path in enumerate(poses, start=1):
        info = window_for_pose(path)
        items.append(info)
        flag = f"  bursts={info['bursts']}" if info["bursts"] > 1 else ""
        extra = f"  {info['fallback']}" if info["fallback"] else ""
        print(
            f"[{index}/{len(poses)}] {path.stem}: "
            f"{info['start']}:{info['end']} / {info['frames']}  "
            f"({info['kept_frac']*100:.0f}% {info['start_sec']:.2f}-{info['end_sec']:.2f}s)"
            f"{flag}{extra}"
        )

    kept = np.array([item["kept_frac"] for item in items], dtype=np.float32)
    multi = sum(1 for item in items if item["bursts"] > 1)
    fallback = sum(1 for item in items if item["fallback"])
    summary = {
        "total": len(items),
        "median_kept": round(float(np.median(kept)), 4),
        "mean_kept": round(float(np.mean(kept)), 4),
        "min_kept": round(float(np.min(kept)), 4),
        "multi_burst": multi,
        "fallback": fallback,
        "pre_roll": PRE_ROLL,
        "post_roll": POST_ROLL,
        "pause_seconds": PAUSE_SECONDS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary)
    print(f"reporte: {args.output}")


if __name__ == "__main__":
    main()
