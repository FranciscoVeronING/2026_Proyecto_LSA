"""Recorta la vista frontal de videos mosaico del Señario (2x2 izquierdo).

En esos clips hay un recuadro grande al frente y dos tomas chicas a la derecha.
Este script detecta la franja negra entre ambos y se queda con el panel grande.

  py -3.11 extraction/crop_mosaic.py --video extraction/out/videos/A.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "out" / "videos"
OUT_DIR = ROOT / "out"


def sample_gray_columns(path: Path, n_samples: int = 7) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cols = []
    rows = []
    picks = np.linspace(0, max(total - 1, 0), n_samples, dtype=int)
    mid = None
    for index in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cols.append(gray.mean(axis=0))
        rows.append(gray.mean(axis=1))
        if mid is None:
            mid = frame
    cap.release()
    if not cols:
        raise RuntimeError(f"No se pudo leer {path}")
    return np.mean(cols, axis=0), np.mean(rows, axis=0), width, height, fps


def longest_dark_run(values: np.ndarray, ceiling: float, min_width: int) -> tuple[int, int] | None:
    dark = values < ceiling
    best: tuple[int, int] | None = None
    start = None
    for i, is_dark in enumerate(dark):
        if is_dark and start is None:
            start = i
        elif not is_dark and start is not None:
            if i - start >= min_width and (best is None or i - start > best[1] - best[0]):
                best = (start, i)
            start = None
    if start is not None and len(dark) - start >= min_width:
        if best is None or len(dark) - start > best[1] - best[0]:
            best = (start, len(dark))
    return best


def detect_main_panel(path: Path) -> dict:
    col, row, width, height, fps = sample_gray_columns(path)
    xs = np.where(col > 12)[0]
    ys = np.where(row > 12)[0]
    if len(xs) < 50 or len(ys) < 50:
        return {"mosaic": False, "reason": "sin contenido", "fps": fps, "width": width, "height": height}
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    zone = col[x0:x1]
    search_from = int(len(zone) * 0.55)
    gutter_rel = longest_dark_run(zone[search_from:], ceiling=8.0, min_width=6)
    if gutter_rel is None:
        return {
            "mosaic": False,
            "reason": "no hay franja negra a la derecha",
            "fps": fps,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
        }
    gutter0 = x0 + search_from + gutter_rel[0]
    gutter1 = x0 + search_from + gutter_rel[1]
    main_w = gutter0 - x0
    pip_w = x1 - gutter1
    if main_w < 400 or pip_w < 120 or main_w < pip_w:
        return {"mosaic": False, "reason": "proporciones raras", "fps": fps, "gutter": (gutter0, gutter1)}
    return {
        "mosaic": True,
        "fps": fps,
        "width": width,
        "height": height,
        "x0": x0,
        "x1": gutter0,
        "y0": y0,
        "y1": y1,
        "gutter": (gutter0, gutter1),
        "pip": (gutter1, x1, y0, y1),
    }


def crop_video(src: Path, dest: Path, box: dict) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = float(box["fps"] or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    x0, x1, y0, y1 = box["x0"], box["x1"], box["y0"], box["y1"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, y1 - y0))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo escribir {dest}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame[y0:y1, x0:x1])
    cap.release()
    writer.release()


def save_overlay(src: Path, dest: Path, box: dict) -> None:
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 3))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    vis = frame.copy()
    cv2.rectangle(vis, (box["x0"], box["y0"]), (box["x1"] - 1, box["y1"] - 1), (0, 220, 80), 4)
    gx0, gx1 = box["gutter"]
    cv2.rectangle(vis, (gx1, box["y0"]), (box["pip"][1] - 1, box["y1"] - 1), (40, 40, 220), 3)
    cv2.putText(vis, "vista principal", (box["x0"] + 16, box["y0"] + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 80), 2)
    cv2.putText(vis, "otras camaras", (gx1 + 8, box["y0"] + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), vis)


def resolve_video(path: Path) -> Path:
    if path.exists():
        return path
    candidate = (ROOT / path).resolve() if not path.is_absolute() else path
    if candidate.exists():
        return candidate
    return VIDEO_DIR / path.name


def process_one(video: Path, output: Path | None, overlay_dir: Path | None, in_place: bool) -> dict:
    box = detect_main_panel(video)
    result = {"file": video.name, "stem": video.stem, "mosaic": bool(box.get("mosaic")), "reason": box.get("reason")}
    if box.get("mosaic"):
        result["size"] = f"{box['x1']-box['x0']}x{box['y1']-box['y0']}"
        result["box"] = {"x0": box["x0"], "x1": box["x1"], "y0": box["y0"], "y1": box["y1"]}
    if not box.get("mosaic"):
        result["action"] = "skip"
        return result
    dest = output or video
    tmp = video.with_name(video.stem + ".__crop__.mp4") if in_place else dest
    if tmp.exists():
        tmp.unlink()
    if overlay_dir is not None:
        save_overlay(video, overlay_dir / f"{video.stem}_crop.jpg", box)
    crop_video(video, tmp, box)
    if in_place:
        tmp.replace(video)
        dest = video
    result["action"] = "cropped"
    result["output"] = str(dest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Recorre extraction/out/videos/")
    parser.add_argument("--in-place", action="store_true", help="Reemplaza el mp4 original por el recorte.")
    parser.add_argument("--dry-run", action="store_true", help="Solo detecta, no escribe.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overlays", action="store_true", help="Guarda JPG con el recuadro verde/rojo.")
    parser.add_argument(
        "--from-report",
        type=Path,
        default=None,
        help="Solo procesa los mosaic=true de un mosaic_crop_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay_dir = (OUT_DIR / "layout_preview") if args.overlays else None
    if args.from_report:
        payload = json.loads(args.from_report.read_text(encoding="utf-8"))
        videos = []
        for item in payload.get("items", []):
            if not item.get("mosaic"):
                continue
            path = VIDEO_DIR / item["file"]
            if path.exists():
                videos.append(path)
    elif args.all:
        videos = sorted(VIDEO_DIR.glob("*.mp4"))
    else:
        videos = [resolve_video(args.video or VIDEO_DIR / "A.mp4")]

    report: list[dict] = []
    cropped = skipped = errors = 0
    for index, video in enumerate(videos, start=1):
        try:
            if args.dry_run:
                box = detect_main_panel(video)
                is_mosaic = bool(box.get("mosaic"))
                row = {"file": video.name, "mosaic": is_mosaic, "reason": box.get("reason")}
                if is_mosaic:
                    row["size"] = f"{box['x1']-box['x0']}x{box['y1']-box['y0']}"
                    cropped += 1
                    print(f"[{index}/{len(videos)}] MOSAICO {video.stem}  {row['size']}")
                else:
                    skipped += 1
                    print(f"[{index}/{len(videos)}] plano   {video.stem}")
                report.append(row)
                continue
            output = args.output if (args.output and not args.all) else None
            in_place = args.in_place or args.all or bool(args.from_report)
            row = process_one(video, output, overlay_dir, in_place=in_place)
            if row.get("action") == "cropped":
                cropped += 1
                print(f"[{index}/{len(videos)}] recorte {video.stem}  {row.get('size')}")
            else:
                skipped += 1
                print(f"[{index}/{len(videos)}] plano   {video.stem}")
            report.append(row)
        except Exception as error:
            errors += 1
            print(f"[{index}/{len(videos)}] ERROR {video.name}: {error}")
            report.append({"file": video.name, "action": "error", "error": str(error)})

    summary = {"total": len(videos), "cropped": cropped, "skipped": skipped, "errors": errors}
    print(summary)
    report_path = OUT_DIR / "mosaic_crop_report.json"
    report_path.write_text(
        json.dumps({"summary": summary, "items": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"reporte: {report_path}")


if __name__ == "__main__":
    main()
