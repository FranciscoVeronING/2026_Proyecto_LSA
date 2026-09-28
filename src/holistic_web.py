"""Cliente Python del Holistic JS/WASM (@mediapipe/holistic, GPU/WebGL)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

import config as cfg

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holistic_web")
SERVER_JS = os.path.join(DIR, "extract_server.mjs")
HOLISTIC_JS = os.path.join(DIR, "node_modules", "@mediapipe", "holistic", "holistic.js")
INBOX_DIR = os.path.join(DIR, "inbox")
JPEG_QUALITY = 80
MAX_LONG_SIDE = 1280


class HolisticWebError(RuntimeError):
    pass


def is_session_dead(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "target crashed",
        "target closed",
        "has been closed",
        "browser has been closed",
        "page crashed",
        "stdin cerrado",
        "se cerró",
        "sin respuesta",
        "broken pipe",
        "connection closed",
        "acceso denegado",
        "winerror 5",
        "permission denied",
    )
    return any(n in text for n in needles)


def _which_node() -> str:
    node = shutil.which("node")
    if not node:
        raise HolisticWebError(
            "No encuentro Node.js en PATH. Instalá Node 18+ y volvé a intentar."
        )
    return node


def _which_npm() -> str:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        raise HolisticWebError("No encuentro npm en PATH.")
    return npm


def ensure_installed() -> None:
    if not os.path.isfile(SERVER_JS):
        raise HolisticWebError(f"Falta {SERVER_JS}")
    if os.path.isfile(HOLISTIC_JS):
        return
    npm = _which_npm()
    print("[*] Instalando @mediapipe/holistic + Playwright (npm install)...")
    subprocess.check_call([npm, "install"], cwd=DIR)
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx:
        subprocess.check_call([npx, "playwright", "install", "chromium"], cwd=DIR)
    if not os.path.isfile(HOLISTIC_JS):
        raise HolisticWebError(
            f"npm install no dejó {HOLISTIC_JS}. "
            f"Correr: cd src/holistic_web && npm install"
        )


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_LONG_SIDE:
        return frame
    scale = MAX_LONG_SIDE / float(long_side)
    return cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class HolisticWebSession:
    """Proceso Node persistente: un Chrome con WebGL GPU + Holistic 0.5."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        require_gpu: bool = True,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.require_gpu = require_gpu
        self.proc: Optional[subprocess.Popen] = None
        self.gpu_info: dict[str, Any] = {}
        self._videos_since_recycle = 0
        self._frame_id = 0
        self._prev_jpeg: Optional[str] = None
        ensure_installed()
        os.makedirs(INBOX_DIR, exist_ok=True)
        self._start()

    def _start(self) -> None:
        env = os.environ.copy()
        env["HOLISTIC_WEB_REQUIRE_GPU"] = "1" if self.require_gpu else "0"
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [_which_node(), SERVER_JS],
            cwd=DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            ready = self._read()
        except Exception:
            self.close()
            raise
        if ready.get("error"):
            self.close()
            raise HolisticWebError(ready["error"])
        if not ready.get("ok"):
            self.close()
            raise HolisticWebError(f"Holistic Web no arrancó: {ready}")
        self.gpu_info = ready.get("gpu") or {}
        renderer = str(self.gpu_info.get("renderer") or "")
        print(
            f"[*] MediaPipe Web {cfg.MEDIAPIPE_WEB_ID} | "
            f"GPU: {self.gpu_info.get('vendor', '?')} / {renderer}"
        )
        ack = self.request(
            {
                "cmd": "init",
                "minDetectionConfidence": self.min_detection_confidence,
                "minTrackingConfidence": self.min_tracking_confidence,
            }
        )
        if ack.get("error"):
            self.close()
            raise HolisticWebError(ack["error"])
        self._videos_since_recycle = 0

    def recycle(self, reason: str = "preventivo") -> None:
        print(f"[*] Reiniciando Chrome/Holistic ({reason})...")
        self.close()
        time.sleep(1.5)
        self._start()

    def maybe_recycle(self, every_videos: int = 30) -> None:
        self._videos_since_recycle += 1
        if self._videos_since_recycle >= every_videos:
            self.recycle(f"cada {every_videos} videos")

    def _drain_stderr(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for raw in self.proc.stderr:
            sys.stderr.write(raw.decode("utf-8", errors="replace"))
            sys.stderr.flush()

    def _write(self, payload: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise HolisticWebError("stdin cerrado")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise HolisticWebError("El proceso Holistic Web se cerró.") from exc

    def _read(self) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise HolisticWebError("stdout cerrado")
        raw = self.proc.stdout.readline()
        if not raw:
            code = self.proc.poll()
            raise HolisticWebError(
                f"Holistic Web terminó sin respuesta (exit={code}). "
                "Revisá stderr: hace falta Chrome con WebGL GPU."
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HolisticWebError(f"Respuesta inválida: {raw[:200]!r}") from exc

    def request(self, payload: dict) -> dict:
        self._write(payload)
        msg = self._read()
        if msg.get("error"):
            raise HolisticWebError(str(msg["error"]))
        return msg

    def process_bgr(self, frame: np.ndarray) -> dict:
        frame = _downscale(frame)
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            raise HolisticWebError("No se pudo encodear el frame a JPEG.")
        self._frame_id += 1
        name = f"f{self._frame_id}.jpg"
        path = os.path.join(INBOX_DIR, name)
        with open(path, "wb") as fh:
            fh.write(buf.tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        try:
            msg = self.request({"cmd": "frame", "file": name})
        finally:
            if self._prev_jpeg and self._prev_jpeg != path:
                try:
                    os.remove(self._prev_jpeg)
                except OSError:
                    pass
            self._prev_jpeg = path
        return {
            "pose": msg.get("pose"),
            "leftHand": msg.get("leftHand"),
            "rightHand": msg.get("rightHand"),
            "face": msg.get("face"),
        }

    def reset(self) -> None:
        self.request({"cmd": "reset"})

    def close(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b'{"cmd":"close"}\n')
                self.proc.stdin.flush()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def __enter__(self) -> "HolisticWebSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def mediapipe_label() -> str:
    return cfg.MEDIAPIPE_WEB_ID
