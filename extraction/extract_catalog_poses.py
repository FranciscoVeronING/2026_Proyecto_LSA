"""Paso 2 del diccionario: catálogo JSON → mp4 YouTube → archivo .pose crudo.

Usa pose-format + MediaPipe Holistic clásico. Corre en **Python 3.11**
(en 3.13 MediaPipe 1.0 ya no expone `mp.solutions.holistic`).

Por cada seña:

1. Descarga un mp4 único con yt-dlp (sin fusionar audio; no hace falta FFmpeg).
   Cache en `extraction/out/videos/`. Si existe, no vuelve a bajar.
2. `pose_video(..., "mediapipe")` estima Holistic en **todos** los frames.
   Coordenadas en píxeles de imagen; no normaliza por hombros ni recorta a 32.
   El z de cada mano es relativo a la muñeca y ~10× más chico que XY: Holistic
   no da 3D métrico de las 21 juntas (POSE_WORLD solo tiene las 33 del cuerpo).
   El recorte de reposo (`sign_windows`) no se aplica acá: el `.pose` queda
   crudo; el visor / export JSON recortan el inicio y el fin útiles.
3. Valida: ≥5 frames, media |coord| > 1, ≥20 % de pose con confianza > 0.3.
   Si falla, borra el `.pose` (no se persisten ceros).
4. Resume: si el `.pose` ya es válido, lo saltea. `--force` reprocesa.

Salida: `extraction/out/poses/<glosa>.pose` (el nombre es el del archivo en disco,
sin `vol1_`/`vol2_`) y `extract_report.json`.

  py -3.11 extraction/extract_catalog_poses.py --only HOLA,MAMÁ
  py -3.11 extraction/extract_catalog_poses.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sign_names import catalog_disk_stem, find_cached_video, fold_spaces
CATALOG = ROOT / "out" / "qr_catalog.json"
VIDEO_DIR = ROOT / "out" / "videos"
POSE_DIR = ROOT / "out" / "poses"
REPORT = ROOT / "out" / "extract_report.json"

SAFE_CHARS = re.compile(r'[<>:"/\\|?*]+')


def load_catalog() -> list[dict]:
    """Filas `kind=sign` con URL. Playlist y notas culturales se ignoran."""
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("kind") == "sign" and row.get("url")]


def stem_for(row: dict) -> str:
    """Nombre en disco: `HOLA`, `F`, `Agua`. Sin prefijo de volumen."""
    return catalog_disk_stem(row)


def cookies_file() -> str | None:
    """Ruta a cookies.txt si existe (videos privados / age-gate de YouTube)."""
    for candidate in (ROOT / "cookies.txt", ROOT.parent / "cookies.txt"):
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
    return None


def download_video(url: str, dest_stem: Path) -> Path:
    """Baja un solo mp4. Formato `best[ext=mp4]` evita el merge audio+video."""
    import yt_dlp

    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    cached = find_cached_video(dest_stem.name)
    if cached:
        return cached
    existing = list(dest_stem.parent.glob(dest_stem.name + ".*"))
    existing = [p for p in existing if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".m4v"}]
    if existing:
        return existing[0]

    options = {
        "format": "best[ext=mp4]/bestvideo[ext=mp4]/best",
        "outtmpl": str(dest_stem) + ".%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    cookie = cookies_file()
    if cookie:
        options["cookiefile"] = cookie

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([url])

    existing = list(dest_stem.parent.glob(dest_stem.name + ".*"))
    existing = [p for p in existing if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".m4v"}]
    if not existing:
        raise RuntimeError(f"yt-dlp no dejó video para {url}")
    return existing[0]


def pose_is_valid(pose_path: Path) -> tuple[bool, dict]:
    """False si hay pocos frames, coordenadas ~0 o casi nada de pose visible."""
    from pose_format import Pose

    data = pose_path.read_bytes()
    pose = Pose.read(data)
    body = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    frames = int(body.shape[0])
    abs_mean = float(np.mean(np.abs(body)))
    conf_mean = float(np.mean(conf)) if conf.size else 0.0
    pose_conf = conf[..., :33] if conf.shape[-1] >= 33 else conf
    pose_hit = float(np.mean(pose_conf > 0.3)) if pose_conf.size else 0.0
    info = {
        "frames": frames,
        "abs_mean": round(abs_mean, 4),
        "conf_mean": round(conf_mean, 4),
        "pose_hit": round(pose_hit, 4),
        "shape": list(body.shape),
        "fps": float(getattr(pose.body, "fps", 0) or 0),
    }
    ok = frames >= 5 and abs_mean > 1.0 and pose_hit >= 0.2
    return ok, info


def extract_pose(video_path: Path, pose_path: Path) -> None:
    """Holistic complejidad 1 + smooth_landmarks. Un worker: más estable en Windows."""
    from pose_format.bin.pose_estimation import pose_video

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_video(
        str(video_path),
        str(pose_path),
        "mediapipe",
        additional_config={"model_complexity": 1, "smooth_landmarks": True},
        progress=False,
        pose_workers=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", type=str, default="", help="ids separados por coma, ej. HOLA,MAMÁ")
    parser.add_argument("--volume", type=str, default="", help="vol1 o vol2")
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reprocesa aunque el .pose ya exista.")
    parser.add_argument(
        "--mosaic-report",
        type=Path,
        default=None,
        help="Re-extrae .pose de los mp4 recortados listados en mosaic_crop_report.json (sin YouTube).",
    )
    return parser.parse_args()


def wanted(row: dict, args: argparse.Namespace) -> bool:
    """Filtro `--volume vol1` y `--only HOLA,F` (id, label o nombre de archivo)."""
    if args.volume and row.get("volume") != args.volume:
        return False
    if not args.only:
        return True
    needles = {item.strip().lower() for item in args.only.split(",") if item.strip()}
    ident = str(row.get("id", "")).strip().lower()
    label = str(row.get("label", "")).strip().lower()
    if ident in needles or label in needles:
        return True
    disk = fold_spaces(stem_for(row))
    folded_needles = {fold_spaces(item) for item in needles}
    return disk in folded_needles


def mosaic_videos(report_path: Path) -> list[Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    videos = []
    for item in payload.get("items", []):
        if not (item.get("mosaic") or item.get("action") == "cropped"):
            continue
        name = item.get("file") or f"{item.get('stem','')}.mp4"
        path = VIDEO_DIR / name
        if path.exists():
            videos.append(path)
    return videos


def extract_local_videos(videos: list[Path], force: bool, keep_invalid: bool) -> int:
    report = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(videos),
        "ok": [],
        "failed": [],
        "skipped": [],
        "source": "local-videos",
    }
    print(f"Señas a procesar (video local): {len(videos)}")
    for index, video_path in enumerate(videos, start=1):
        stem = video_path.stem
        pose_path = POSE_DIR / f"{stem}.pose"
        print(f"\n[{index}/{len(videos)}] {stem}")
        if pose_path.exists() and not force:
            ok, info = pose_is_valid(pose_path)
            if ok:
                print(f"  ya existe y es válido: {info}")
                report["skipped"].append({"id": stem, **info})
                continue
            print(f"  .pose inválido, se reintenta: {info}")
            pose_path.unlink(missing_ok=True)
        try:
            print(f"  video: {video_path.name}")
            extract_pose(video_path, pose_path)
            ok, info = pose_is_valid(pose_path)
            if not ok:
                raise RuntimeError(f"landmarks vacíos o sin pose: {info}")
            print(f"  ok: {info}")
            report["ok"].append({"id": stem, **info})
        except Exception as error:
            print(f"  ERROR: {error}")
            if pose_path.exists() and not keep_invalid:
                pose_path.unlink(missing_ok=True)
            report["failed"].append({"id": stem, "error": str(error)})
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== resumen ====")
    print(f"ok={len(report['ok'])} skipped={len(report['skipped'])} failed={len(report['failed'])}")
    print(f"reporte: {REPORT}")
    return 0 if not report["failed"] else 1


def main() -> int:
    args = parse_args()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    if args.mosaic_report:
        videos = mosaic_videos(args.mosaic_report)
        if args.limit:
            videos = videos[: args.limit]
        return extract_local_videos(videos, force=True, keep_invalid=args.keep_invalid)

    rows = [row for row in load_catalog() if wanted(row, args)]
    if args.limit:
        rows = rows[: args.limit]

    report = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "ok": [],
        "failed": [],
        "skipped": [],
    }

    print(f"Señas a procesar: {len(rows)}")
    print(f"Cookies: {cookies_file() or 'no (yt-dlp sin cookies.txt)'}")

    for index, row in enumerate(rows, start=1):
        stem = stem_for(row)
        pose_path = POSE_DIR / f"{stem}.pose"
        print(f"\n[{index}/{len(rows)}] {stem}  {row['url']}")

        if pose_path.exists() and not args.force:
            ok, info = pose_is_valid(pose_path)
            if ok:
                print(f"  ya existe y es válido: {info}")
                report["skipped"].append({"id": stem, **info})
                continue
            print(f"  .pose inválido, se reintenta: {info}")
            pose_path.unlink(missing_ok=True)

        try:
            video_path = download_video(row["url"], VIDEO_DIR / stem)
            print(f"  video: {video_path.name}")
            extract_pose(video_path, pose_path)
            ok, info = pose_is_valid(pose_path)
            if not ok:
                raise RuntimeError(f"landmarks vacíos o sin pose: {info}")
            print(f"  ok: {info}")
            report["ok"].append({"id": stem, "url": row["url"], **info})
        except Exception as error:
            print(f"  ERROR: {error}")
            if pose_path.exists() and not args.keep_invalid:
                pose_path.unlink(missing_ok=True)
            report["failed"].append({"id": stem, "url": row["url"], "error": str(error)})

        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== resumen ====")
    print(f"ok={len(report['ok'])} skipped={len(report['skipped'])} failed={len(report['failed'])}")
    print(f"reporte: {REPORT}")
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
