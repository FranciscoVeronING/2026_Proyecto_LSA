"""Fit mosaic and full-frame Señario clips into one waist-up landmark canvas.

Two source layouts exist:

* mosaic — 2x2 dictionary video, already cropped to the large front panel
  (~860x1030, size varies a few dozen pixels per sign).
* plano — single camera, usually 1920x1080, signer smaller in the frame.

This does **not** overwrite raw ``out/poses/*.pose``. ``export_pose_json``
applies the same affine when writing viewer clips, so Holistic is not rerun.

The person is centered on the median shoulder midpoint; shoulder width is
mapped to a fixed pixel span. Output canvas is 720x960 (3:4).

  py -3.11 extraction/canonicalize_landmarks.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
POSE_DIR = ROOT / "out" / "poses"
VIDEO_DIR = ROOT / "out" / "videos"

CANON_WIDTH = 720
CANON_HEIGHT = 960
CANON_SHOULDER_MID = (360.0, 400.0)
CANON_SHOULDER_SPAN = 230.0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
IMAGE_COMPONENTS = 33 + 468 + 21 + 21  # pose + face + hands; not world


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def measure_shoulders(xy: np.ndarray, conf: np.ndarray) -> dict | None:
    """xy (T, 33, 2) in original pixels."""
    spans: list[float] = []
    mids_x: list[float] = []
    mids_y: list[float] = []
    for t in range(xy.shape[0]):
        if float(conf[t, LEFT_SHOULDER]) < 0.3 or float(conf[t, RIGHT_SHOULDER]) < 0.3:
            continue
        left = xy[t, LEFT_SHOULDER]
        right = xy[t, RIGHT_SHOULDER]
        span = float(np.hypot(left[0] - right[0], left[1] - right[1]))
        if span < 20.0:
            continue
        spans.append(span)
        mids_x.append(float((left[0] + right[0]) * 0.5))
        mids_y.append(float((left[1] + right[1]) * 0.5))
    span = _median_or_none(spans)
    mx = _median_or_none(mids_x)
    my = _median_or_none(mids_y)
    if span is None or mx is None or my is None:
        return None
    scale = CANON_SHOULDER_SPAN / span
    return {
        "scale": scale,
        "tx": CANON_SHOULDER_MID[0] - mx * scale,
        "ty": CANON_SHOULDER_MID[1] - my * scale,
        "src_span": span,
        "src_mid": (mx, my),
        "frames_used": len(spans),
    }


def affine_from_pose(pose) -> dict | None:
    from pose_format import Pose  # noqa: F401

    data = np.ma.filled(pose.body.data, 0.0)[:, 0]
    conf = np.ma.filled(pose.body.confidence, 0.0)[:, 0]
    xy = data[:, :33, :2].copy()
    return measure_shoulders(xy, conf[:, :33])


def apply_affine_xy(xyz: np.ndarray, transform: dict, point_count: int) -> None:
    scale = transform["scale"]
    tx = transform["tx"]
    ty = transform["ty"]
    sl = slice(0, point_count)
    xyz[:, sl, 0] = xyz[:, sl, 0] * scale + tx
    xyz[:, sl, 1] = xyz[:, sl, 1] * scale + ty
    xyz[:, sl, 2] = xyz[:, sl, 2] * scale


def canonicalize_pose(pose):
    """Mutate `pose` so image landmarks live on the 720x960 canvas. Raw files stay untouched."""
    from pose_format.numpy import NumPyPoseBody
    from pose_format.pose_header import PoseHeaderDimensions
    import numpy.ma as ma

    transform = affine_from_pose(pose)
    if transform is None:
        return pose, None
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    apply_affine_xy(data[:, 0], transform, min(IMAGE_COMPONENTS, data.shape[2]))
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    pose.header.dimensions = PoseHeaderDimensions(
        CANON_WIDTH,
        CANON_HEIGHT,
        pose.header.dimensions.depth,
    )
    return pose, transform


def affine_matrix(transform: dict) -> np.ndarray:
    scale = transform["scale"]
    return np.array([[scale, 0.0, transform["tx"]], [0.0, scale, transform["ty"]]], dtype=np.float32)


def warp_frame(frame: np.ndarray, transform: dict) -> np.ndarray:
    import cv2

    return cv2.warpAffine(
        frame,
        affine_matrix(transform),
        (CANON_WIDTH, CANON_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print affine stats, do not write.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from pose_format import Pose

    args = parse_args()
    paths = sorted(POSE_DIR.glob("*.pose"), key=lambda p: p.stem.casefold())
    if args.limit:
        paths = paths[: args.limit]
    rows = []
    for path in paths:
        pose = Pose.read(path.read_bytes())
        src_w = float(pose.header.dimensions.width)
        src_h = float(pose.header.dimensions.height)
        transform = affine_from_pose(pose)
        kind = "mosaic-like" if src_h >= src_w else "plano"
        row = {
            "stem": path.stem,
            "kind": kind,
            "src": f"{int(src_w)}x{int(src_h)}",
            "ok": transform is not None,
        }
        if transform:
            row["scale"] = round(transform["scale"], 4)
            row["frames_used"] = transform["frames_used"]
        rows.append(row)
        print(f"{path.stem:28} {kind:12} {row['src']:12} scale={row.get('scale', '-')}")
    summary = {
        "total": len(rows),
        "ok": sum(1 for r in rows if r["ok"]),
        "mosaic_like": sum(1 for r in rows if r["kind"] == "mosaic-like"),
        "plano": sum(1 for r in rows if r["kind"] == "plano"),
        "canon": f"{CANON_WIDTH}x{CANON_HEIGHT}",
    }
    print(summary)
    report = ROOT / "out" / "canonicalize_report.json"
    report.write_text(json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reporte: {report}")


if __name__ == "__main__":
    main()
