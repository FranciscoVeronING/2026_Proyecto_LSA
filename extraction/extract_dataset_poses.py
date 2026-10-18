"""Extrae .pose crudos de señas del dataset/ que todavía no están en out/poses.

Por cada carpeta de dataset sin equivalente en el diccionario: toma el video
`*-1.mp4` (si no hay, el primero que no sea 'copia'), corre Holistic en todos
los frames y guarda `extraction/out/poses/<carpeta>.pose`.

  py -3.11 extraction/extract_dataset_poses.py
  py -3.11 extraction/extract_dataset_poses.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from extract_catalog_poses import POSE_DIR, extract_pose, pose_is_valid
from sign_names import fold_loose, fold_spaces, sanitize_stem

DATASET = ROOT.parent / "dataset"
REPORT = ROOT / "out" / "extract_dataset_report.json"
RETRY_REPORT = ROOT / "out" / "extract_dataset_retry.json"
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".m4v", ".avi", ".mov"}
LEFT_HAND_START = 33 + 468
HAND_POINTS = 42

# El dataset nombra dígitos; el Señario usa la glosa de número.
DIGIT_TO_GLOSS = {
    "0": "CERO",
    "1": "UNO",
    "2": "DOS",
    "3": "TRES",
    "4": "CUATRO",
    "5": "CINCO",
    "6": "SEIS",
    "7": "SIETE",
    "8": "OCHO",
    "9": "NUEVE",
}


def pose_index() -> dict[str, list[str]]:
    by_loose: dict[str, list[str]] = {}
    for path in sorted(POSE_DIR.glob("*.pose")):
        by_loose.setdefault(fold_loose(path.stem), []).append(path.stem)
    return by_loose


def loose_keys(name: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    variants = [
        name,
        name.replace("_", " ").replace(".", " "),
        name.replace("_", ""),
        name.replace("_a", " a"),
        name.replace("_En", " En"),
    ]
    parts = [part for part in re.split(r"[_\s.]+", name) if part]
    if len(parts) == 2:
        variants.append(f"{parts[1]} {parts[0]}")
        variants.append(f"{parts[1]}_{parts[0]}")
    if name in DIGIT_TO_GLOSS:
        variants.append(DIGIT_TO_GLOSS[name])
    for variant in variants:
        key = fold_loose(variant)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def match_existing(name: str, by_loose: dict[str, list[str]]) -> str | None:
    hits: list[str] = []
    for key in loose_keys(name):
        hits.extend(by_loose.get(key, []))
    hits = list(dict.fromkeys(hits))
    if not hits:
        return None
    exact = [stem for stem in hits if fold_loose(stem) == fold_loose(name)]
    if len(exact) == 1:
        return exact[0]
    if name == "Papa":
        for stem in hits:
            if fold_loose(stem) == "papa" and stem != "Papa":
                return stem
    if len(hits) == 1:
        return hits[0]
    if name in DIGIT_TO_GLOSS:
        want = fold_loose(DIGIT_TO_GLOSS[name])
        for stem in hits:
            if fold_loose(stem) == want:
                return stem
    return hits[0]


def list_videos(folder: Path) -> list[Path]:
    videos = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and "copia" not in path.name.lower()
    ]
    if not videos:
        raise FileNotFoundError(f"sin video en {folder}")

    def clip_num(path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem.replace(" ", ""))
        return int(match.group(1)) if match else 10**9

    return sorted(videos, key=clip_num)


def pick_video(folder: Path) -> Path:
    videos = list_videos(folder)
    ones = [path for path in videos if re.search(r"[-_.]1$", path.stem)]
    return (ones or videos)[0]


def sample_videos(folder: Path, tries: int) -> list[Path]:
    videos = list_videos(folder)
    if len(videos) <= tries:
        return videos
    picks = [videos[0]]
    inner = videos[1:]
    slots = max(tries - 1, 1)
    for index in range(slots):
        pos = int(round((index + 1) / (slots + 1) * (len(inner) - 1)))
        candidate = inner[pos]
        if candidate not in picks:
            picks.append(candidate)
    return picks


def score_pose(pose_path: Path) -> dict:
    from pose_format import Pose

    pose = Pose.read(pose_path.read_bytes())
    body = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    frames = int(body.shape[0])
    t0 = int(round(frames * 0.25))
    t1 = int(round(frames * 0.75))
    mid = slice(t0, max(t1, t0 + 1))
    pose_conf = conf[mid, 0, :33] if conf.shape[-1] >= 33 else conf[mid]
    pose_hit = float(np.mean(pose_conf > 0.3)) if pose_conf.size else 0.0
    hand_end = min(LEFT_HAND_START + HAND_POINTS, body.shape[2])
    if body.shape[2] > LEFT_HAND_START:
        hand_xy = body[mid, 0, LEFT_HAND_START:hand_end, :2]
        hand_conf = conf[mid, 0, LEFT_HAND_START:hand_end]
        visible = (hand_conf >= 0.3) & (np.linalg.norm(hand_xy, axis=-1) >= 8.0)
        hand_hit = float(np.mean(visible)) if visible.size else 0.0
        if hand_xy.shape[0] > 1:
            motion = float(np.mean(np.linalg.norm(np.diff(hand_xy, axis=0), axis=-1)))
        else:
            motion = 0.0
    else:
        hand_hit = 0.0
        motion = 0.0
    return {
        "frames": frames,
        "pose_hit": round(pose_hit, 4),
        "hand_hit": round(hand_hit, 4),
        "hand_motion": round(motion, 4),
        "score": round(3.0 * hand_hit + pose_hit + min(motion, 40.0) / 80.0, 4),
    }


def find_dataset_folder(name: str) -> Path:
    want = fold_loose(name)
    for folder in DATASET.iterdir():
        if folder.is_dir() and fold_loose(folder.name) == want:
            return folder
    raise FileNotFoundError(f"no está dataset/{name}")


def retry_best(names: list[str], tries: int, keep_invalid: bool) -> int:
    tmp_dir = POSE_DIR / "_retry"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tries": tries,
        "ok": [],
        "failed": [],
    }
    for index, name in enumerate(names, start=1):
        folder = find_dataset_folder(name)
        stem = pose_stem_for(folder.name)
        dest = POSE_DIR / f"{stem}.pose"
        candidates = sample_videos(folder, tries)
        print(f"\n[{index}/{len(names)}] {folder.name}  {len(candidates)} clips")
        ranked: list[dict] = []
        for video_path in candidates:
            tmp = tmp_dir / f"{stem}__{video_path.stem}.pose"
            try:
                if dest.exists() and re.search(r"[-_.]1$", video_path.stem):
                    tmp.write_bytes(dest.read_bytes())
                else:
                    extract_pose(video_path, tmp)
                ok, info = pose_is_valid(tmp)
                if not ok:
                    raise RuntimeError(f"inválido: {info}")
                metrics = score_pose(tmp)
                row = {"video": video_path.name, **metrics}
                ranked.append(row)
                print(
                    f"  {video_path.name}: hand={metrics['hand_hit']:.2f} "
                    f"pose={metrics['pose_hit']:.2f} motion={metrics['hand_motion']:.1f} "
                    f"score={metrics['score']:.2f}"
                )
            except Exception as error:
                print(f"  {video_path.name}: ERROR {error}")
                if tmp.exists() and not keep_invalid:
                    tmp.unlink(missing_ok=True)
        if not ranked:
            report["failed"].append({"id": stem, "error": "ningún clip válido"})
            continue
        ranked.sort(key=lambda row: row["score"], reverse=True)
        winner = ranked[0]
        win_tmp = tmp_dir / f"{stem}__{Path(winner['video']).stem}.pose"
        dest.write_bytes(win_tmp.read_bytes())
        print(f"  -> {winner['video']}")
        report["ok"].append({"id": stem, "chosen": winner, "tried": ranked})
        RETRY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for leftover in tmp_dir.glob(f"{stem}__*.pose"):
            leftover.unlink(missing_ok=True)

    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    print("\n==== retry ====")
    print(f"ok={len(report['ok'])} failed={len(report['failed'])}")
    print(f"reporte: {RETRY_REPORT}")
    return 0 if not report["failed"] else 1


def pose_stem_for(folder_name: str) -> str:
    stem = sanitize_stem(folder_name)
    if not stem:
        raise ValueError(f"nombre vacío: {folder_name!r}")
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument(
        "--retry",
        default="",
        help="Reintenta esas glosas con varios clips y deja el de mejor mano, ej. Brazo,Mio",
    )
    parser.add_argument("--tries", type=int, default=7, help="Cuántos clips probar por seña (default: 7).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retry:
        names = [item.strip() for item in args.retry.split(",") if item.strip()]
        return retry_best(names, max(args.tries, 2), args.keep_invalid)
    if not DATASET.is_dir():
        print(f"No está {DATASET}")
        return 1
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    by_loose = pose_index()
    folders = sorted([path for path in DATASET.iterdir() if path.is_dir()], key=lambda p: p.name.lower())

    already: list[dict] = []
    missing: list[Path] = []
    for folder in folders:
        found = match_existing(folder.name, by_loose)
        if found:
            already.append({"dataset": folder.name, "pose": found})
        else:
            missing.append(folder)

    print(f"dataset={len(folders)} ya cubiertas={len(already)} faltan={len(missing)}")
    if not missing:
        return 0

    print("Faltan:")
    for folder in missing:
        print(f"  {folder.name}")

    report = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "already": already,
        "missing": [folder.name for folder in missing],
        "ok": [],
        "skipped": [],
        "failed": [],
    }
    if args.dry_run:
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"dry-run: {REPORT}")
        return 0

    for index, folder in enumerate(missing, start=1):
        stem = pose_stem_for(folder.name)
        pose_path = POSE_DIR / f"{stem}.pose"
        print(f"\n[{index}/{len(missing)}] {folder.name} -> {pose_path.name}")
        if pose_path.exists() and not args.force:
            ok, info = pose_is_valid(pose_path)
            if ok:
                print(f"  ya existe y es válido: {info}")
                report["skipped"].append({"id": stem, **info})
                continue
            print(f"  .pose inválido, se reintenta: {info}")
            pose_path.unlink(missing_ok=True)
        try:
            video_path = pick_video(folder)
            print(f"  video: {video_path.name}")
            extract_pose(video_path, pose_path)
            ok, info = pose_is_valid(pose_path)
            if not ok:
                raise RuntimeError(f"landmarks vacíos o sin pose: {info}")
            print(f"  ok: {info}")
            report["ok"].append({"id": stem, "video": video_path.name, **info})
        except Exception as error:
            print(f"  ERROR: {error}")
            if pose_path.exists() and not args.keep_invalid:
                pose_path.unlink(missing_ok=True)
            report["failed"].append({"id": stem, "error": str(error)})
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== resumen ====")
    print(f"ok={len(report['ok'])} skipped={len(report['skipped'])} failed={len(report['failed'])}")
    print(f"reporte: {REPORT}")
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
