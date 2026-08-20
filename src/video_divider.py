"""Download a video and split MediaPipe hand-landmark movement into clips.

Install:  pip install opencv-python numpy mediapipe yt-dlp matplotlib
Run:      python VideoDivider.py --preview-roi --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VIDEO_ID = "WuwGGAP-6JI"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x

    @property
    def height(self):
        return self.y2 - self.y


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    peak_change: float

    @property
    def duration(self):
        return self.end_sec - self.start_sec


def parse_args():
    p = argparse.ArgumentParser(
        description="Split hand-landmark movement in a video crop into signal clips."
    )
    p.add_argument("--video", type=Path, help="Local video; default: debate.mp4 next to this script.")
    p.add_argument("--video-id", default=VIDEO_ID, help=f"YouTube ID (default: {VIDEO_ID}).")
    p.add_argument("--start-minute", type=float, default=0, help="Analysis start minute (default: 0).")
    p.add_argument("--duration-minutes", type=float, default=3, help="Duration; 0 means until end (default: 3).")
    p.add_argument("--target-fps", type=float, default=15, help="Landmark analysis FPS (default: 15).")
    p.add_argument("--roi", nargs=4, type=float, default=(0, 0, .40, 1), metavar=("X", "Y", "W", "H"))
    p.add_argument("--select-roi", action="store_true", help="Select the crop interactively.")
    p.add_argument("--output-dir", type=Path, help="Output folder; default: hand_signal_clips.")
    p.add_argument("--full-frame", action="store_true", help="Write full-frame clips instead of cropped clips.")
    p.add_argument("--landmark-threshold", "--motion-threshold", dest="threshold", type=float,
                   help="Fixed start threshold; otherwise it is learned.")
    p.add_argument("--minimum-landmark-change", "--minimum-motion", dest="minimum", type=float, default=.008,
                   help="Minimum automatic threshold (default: .008).")
    p.add_argument("--threshold-sigma", type=float, default=1.0,
                   help="Start-threshold noise multiplier (default: 1.0).")
    p.add_argument("--stop-threshold-sigma", type=float, default=1.5,
                   help="Stop-threshold noise multiplier (default: 1.5).")
    p.add_argument("--stop-landmark-threshold", "--stop-motion-threshold", dest="stop_threshold", type=float,
                   help="Fixed stop threshold.")
    p.add_argument("--pause-seconds", type=float, default=.20,
                   help="Quiet time required to end a signal (default: .20).")
    p.add_argument("--min-signal-seconds", type=float, default=.20,
                   help="Discard shorter detections (default: .20).")
    p.add_argument("--pre-roll", type=float, default=.15, help="Seconds before movement (default: .15).")
    p.add_argument("--post-roll", type=float, default=.20, help="Seconds after movement (default: .20).")
    p.add_argument("--preview-roi", action="store_true", help="Save the crop preview.")
    p.add_argument("--show-preview", action="store_true", help="Display the crop preview window.")
    p.add_argument("--plot", action="store_true", help="Save the landmark-change plot.")
    p.add_argument("--model-path", type=Path, help="Path to hand_landmarker.task.")
    p.add_argument("--num-hands", type=int, default=2, help="Maximum hands to track (default: 2).")
    p.add_argument("--min-hand-detection-confidence", type=float, default=.5)
    p.add_argument("--min-hand-presence-confidence", type=float, default=.5)
    p.add_argument("--min-tracking-confidence", type=float, default=.5)
    return p.parse_args()


def ensure_model(path: Path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 100_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(path))
    except Exception as error:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Download the model manually from {MODEL_URL}") from error


class HandTracker:
    def __init__(self, model: Path, args):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise RuntimeError("Install MediaPipe with: pip install mediapipe") from error

        ensure_model(model)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_hand_detection_confidence,
            min_hand_presence_confidence=args.min_hand_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.mp = mp
        self.task = vision.HandLandmarker.create_from_options(options)
        self.previous = {}
        self.missing = 0

    @staticmethod
    def hands(result):
        found = {}
        for i, landmarks in enumerate(result.hand_landmarks):
            points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            label = ""
            if i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
                label = getattr(category, "category_name", "") or getattr(category, "display_name", "")
            label = label or f"hand_{i}"
            found[f"{label}_{i}" if label in found else label] = points
        return found

    @staticmethod
    def scale(points):
        width, height = np.ptp(points[:, :2], axis=0)
        return max(float(np.hypot(width, height)), 1e-3)

    @classmethod
    def change(cls, old, new):
        displacement = np.linalg.norm(new[:, :2] - old[:, :2], axis=1)
        return float(np.mean(displacement) / max(cls.scale(old), cls.scale(new)))

    def score(self, frame, roi: ROI, timestamp_ms):
        crop = frame[roi.y:roi.y2, roi.x:roi.x2]
        if crop.size == 0:
            raise ValueError("ROI produced an empty crop.")
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        )
        result = self.task.detect_for_video(image, timestamp_ms)
        current = self.hands(result)
        if not current:
            self.missing += 1
            if self.missing > 2:
                self.previous = {}
            return 0.0, 0

        self.missing = 0
        changes, unused = [], set(self.previous)
        for label, points in current.items():
            if label in self.previous:
                changes.append(self.change(self.previous[label], points))
                unused.discard(label)
        for label, points in current.items():
            if label not in self.previous and unused:
                old_label = min(unused, key=lambda k: np.linalg.norm(points[0, :2] - self.previous[k][0, :2]))
                changes.append(self.change(self.previous[old_label], points))
                unused.remove(old_label)
        self.previous = current
        return (max(changes) if changes else 0.0), len(current)

    def close(self):
        self.task.close()


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {path}")
    return cap, fps, frames, width, height


def make_roi(values, width, height):
    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive.")
    x1 = max(0, min(width - 1, round(x * width)))
    y1 = max(0, min(height - 1, round(y * height)))
    x2 = max(x1 + 1, min(width, round((x + w) * width)))
    y2 = max(y1 + 1, min(height, round((y + h) * height)))
    return ROI(x1, y1, x2, y2)


def choose_roi(frame, roi):
    x, y, w, h = cv2.selectROI("Select crop; press ENTER", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select crop; press ENTER")
    return ROI(int(x), int(y), int(x + w), int(y + h)) if w and h else roi


def save_preview(frame, roi, path, show=False):
    preview = frame.copy()
    cv2.rectangle(preview, (roi.x, roi.y), (roi.x2 - 1, roi.y2 - 1), (0, 255, 0), 5)
    cv2.putText(preview, f"ROI {roi.x},{roi.y} {roi.width}x{roi.height}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Could not save preview: {path}")
    if show:
        cv2.imshow("ROI preview", preview)
        cv2.waitKey(0)
        cv2.destroyWindow("ROI preview")


def rolling(values, window, reducer):
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = int(window) | 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray([reducer(padded[i:i + window]) for i in range(len(values))])


def automatic_threshold(scores, minimum, sigma):
    values = scores[scores > 0]
    if not len(values):
        return float(minimum)
    baseline = float(np.percentile(values, 25))
    spread = 1.4826 * float(np.median(np.abs(values - baseline)))
    return max(float(minimum), baseline + sigma * spread)


def detect(scores, frames, fps, start_threshold, stop_threshold, args):
    if len(scores) < 2:
        return []
    step = max(1, int(np.median(np.diff(frames))))
    effective_fps = fps / step
    starts = rolling(scores, 3, np.max)
    stops = rolling(scores, max(3, round(effective_fps / 5)), np.median)
    stop_threshold = max(float(start_threshold), float(stop_threshold))
    pause_n = max(1, math.ceil(args.pause_seconds * effective_fps))
    min_n = max(1, math.ceil(args.min_signal_seconds * effective_fps))
    pre_n = max(0, round(args.pre_roll * effective_fps))
    post_n = max(0, round(args.post_roll * effective_fps))

    runs, active, below = [], None, 0
    for i, (start_value, stop_value) in enumerate(zip(starts, stops)):
        if active is None:
            if start_value >= start_threshold:
                active = i
            continue
        if stop_value < stop_threshold:
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

    segments = []
    for start, end in runs:
        a = max(0, start - pre_n)
        b = min(len(frames) - 1, end - 1 + post_n)
        start_frame = int(frames[a])
        end_frame = int(frames[b]) + step
        segments.append(Segment(start_frame, end_frame, start_frame / fps, end_frame / fps,
                                float(np.max(scores[start:end]))))

    combined = []
    for segment in segments:
        if combined and segment.start_frame <= combined[-1].end_frame:
            old = combined[-1]
            combined[-1] = Segment(old.start_frame, max(old.end_frame, segment.end_frame),
                                   old.start_sec, max(old.end_sec, segment.end_sec),
                                   max(old.peak_change, segment.peak_change))
        else:
            combined.append(segment)
    return combined


def clean_outputs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    patterns = ("signal_*.mp4", "signals.csv", "landmark_scores.csv", "motion_signal_plot.png", "roi_preview.jpg")
    removed = sum(1 for pattern in patterns for path in folder.glob(pattern) if path.is_file() and not path.unlink())
    if removed:
        print(f"Cleaned {removed} previous generated output file(s) from {folder}.")


def timestamp(seconds):
    total = max(0, round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def write_clips(video, segments, roi, folder, crop, fps, total_frames):
    paths, codec = [], cv2.VideoWriter_fourcc(*"mp4v")
    for number, segment in enumerate(segments, 1):
        path = folder / f"signal_{number:04d}_{timestamp(segment.start_sec)}-{timestamp(segment.end_sec)}.mp4"
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
        writer = None; frame_number = segment.start_frame
        try:
            while frame_number < min(segment.end_frame, total_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                out = frame[roi.y:roi.y2, roi.x:roi.x2] if crop else frame
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(str(path), codec, fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create {path}")
                writer.write(out); frame_number += 1
        finally:
            cap.release()
            if writer:
                writer.release()
        if path.exists() and path.stat().st_size:
            paths.append(path)
    return paths


def write_csv(path, segments, clips):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["signal", "start_seconds", "end_seconds", "duration_seconds", "peak_landmark_change", "clip"])
        for i, segment in enumerate(segments):
            writer.writerow([i + 1, f"{segment.start_sec:.3f}", f"{segment.end_sec:.3f}",
                             f"{segment.duration:.3f}", f"{segment.peak_change:.5f}",
                             clips[i].name if i < len(clips) else ""])


def write_scores(path, times, scores):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["video_seconds", "landmark_change"])
        writer.writerows((f"{t:.3f}", f"{s:.6f}") for t, s in zip(times, scores))


def save_plot(path, times, scores, start, stop, segments):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping plot; install matplotlib with: pip install matplotlib")
        return
    plt.figure(figsize=(15, 5)); plt.plot(times / 60, scores, color="purple", label="Landmark change")
    plt.axhline(start, color="green", ls="--", label=f"Start {start:.3f}")
    plt.axhline(stop, color="black", ls=":", label=f"Stop {stop:.3f}")
    for segment in segments:
        plt.axvspan(segment.start_sec / 60, segment.end_sec / 60, color="red", alpha=.18)
    plt.xlabel("Video time (minutes)"); plt.ylabel("Normalized landmark change")
    plt.title("Detected hand-signal intervals"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def download(url, destination):
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("Install yt-dlp with: pip install yt-dlp") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    fmt = "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4]/b" if shutil.which("ffmpeg") else "b[ext=mp4]/b"
    options = {"format": fmt, "outtmpl": str(destination.with_suffix(".%(ext)s")),
               "merge_output_format": "mp4", "noplaylist": True, "retries": 3}
    print(f"Downloading {url} ...")
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    if destination.exists():
        return destination
    for path in sorted(destination.parent.glob(f"{destination.stem}.*")):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return path
    raise FileNotFoundError(f"Download finished but no video was found at {destination}")


def analyze(video, output, args):
    cap, fps, total, width, height = open_video(video)
    tracker = None
    try:
        clean_outputs(output)
        start = max(0, round(args.start_minute * 60 * fps))
        end = min(total - 1, start + round(args.duration_minutes * 60 * fps)) if args.duration_minutes > 0 else total - 1
        if start >= total:
            raise ValueError(f"Start minute is beyond the video ({total / fps / 60:.1f} minutes).")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start); ok, first = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {start}.")
        roi = make_roi(args.roi, width, height)
        if args.select_roi:
            roi = choose_roi(first, roi)
        if args.preview_roi or args.show_preview:
            save_preview(first, roi, output / "roi_preview.jpg", args.show_preview)

        step = max(1, round(fps / max(1, args.target_fps)))
        tracker = HandTracker(args.model_path, args)
        first_score, hands = tracker.score(first, roi, round(start * 1000 / fps))
        frames, times, scores = [start], [start / fps], [first_score]
        frame_number = start; processed = 1
        print(f"Analyzing {video.name}: {start / fps / 60:.2f} to {end / fps / 60:.2f} minutes, "
              f"ROI {roi.width}x{roi.height}, analysis FPS {fps / step:.2f}, first frame hands: {hands}.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if frame_number > end:
                break
            if (frame_number - start) % step:
                continue
            score, _ = tracker.score(frame, roi, round(frame_number * 1000 / fps))
            frames.append(frame_number); times.append(frame_number / fps); scores.append(score); processed += 1
            if processed % max(1, round(fps / step * 30)) == 0:
                print(f"  analyzed {times[-1] / 60:.1f} minutes...")
    finally:
        cap.release()
        if tracker:
            tracker.close()

    scores = np.asarray(scores, dtype=np.float32); frames = np.asarray(frames, dtype=np.int64); times = np.asarray(times, dtype=np.float32)
    write_scores(output / "landmark_scores.csv", times, scores)
    start_threshold = args.threshold if args.threshold is not None else automatic_threshold(scores, args.minimum, args.threshold_sigma)
    stop_threshold = args.stop_threshold if args.stop_threshold is not None else (
        start_threshold if args.threshold is not None else automatic_threshold(scores, args.minimum, args.stop_threshold_sigma)
    )
    stop_threshold = max(start_threshold, stop_threshold)
    segments = detect(scores, frames, fps, start_threshold, stop_threshold, args)
    clips = write_clips(video, segments, roi, output, not args.full_frame, fps, total)
    write_csv(output / "signals.csv", segments, clips)
    if args.plot:
        save_plot(output / "motion_signal_plot.png", times, scores, start_threshold, stop_threshold, segments)
    print("\nResults")
    print(f"  Start threshold: {start_threshold:.5f}")
    print(f"  Stop threshold:  {stop_threshold:.5f}")
    print(f"  Signals detected: {len(segments)}")
    print(f"  Clips written:    {len(clips)}")
    print(f"  Output folder:    {output}")
    print(f"  Manifest:         {output / 'signals.csv'}")


def main():
    args = parse_args(); base = Path(__file__).resolve().parent
    video = args.video or base / "debate.mp4"
    output = args.output_dir or base / "hand_signal_clips"
    args.model_path = args.model_path or base / "hand_landmarker.task"
    if not video.is_absolute():
        video = Path.cwd() / video
    if not output.is_absolute():
        output = Path.cwd() / output
    if not args.model_path.is_absolute():
        args.model_path = Path.cwd() / args.model_path
    if not video.exists():
        video = download(f"https://www.youtube.com/watch?v={args.video_id}", video)
    else:
        print(f"Using existing video: {video}")
    analyze(video, output, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
