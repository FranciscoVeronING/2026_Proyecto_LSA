"""STT local: WAV de Meet → texto con Vosk (español, en esta PC)."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
import urllib.request
import wave

_MODEL_NAME = "vosk-model-small-es-0.42"
_MODEL_URL = f"https://alphacephei.com/vosk/models/{_MODEL_NAME}.zip"
_VOSK_VER = "0.3.45"
_model = None


def transcribe_wav(wav_bytes: bytes, lang: str = "es-ES") -> str:
    del lang
    pcm, rate = _wav_to_pcm16(wav_bytes)
    if rate != 16000:
        pcm = _resample_pcm16(pcm, rate, 16000)
        rate = 16000
    if len(pcm) < 16000 * 0.35 * 2:
        return ""
    from backend.ensure_pkg import ensure_module

    ensure_module("vosk", "vosk==0.3.45")
    _prepare_vosk_native()
    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(_vosk_model(), rate)
    rec.SetWords(False)
    rec.AcceptWaveform(pcm)
    data = json.loads(rec.FinalResult() or "{}")
    return str(data.get("text") or "").strip()


def _frozen_internal() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal"))
    return Path()


def _prepare_vosk_native() -> None:
    """En el exe, Vosk busca DLLs en _internal/vosk. Si PyInstaller no las copió, las bajamos."""
    dest = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ILSA" / "vosk-native"
    if getattr(sys, "frozen", False):
        dest = _frozen_internal() / "vosk"
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "libvosk.dll").is_file():
        print("[backend] STT: bajando libvosk (una vez)…")
        _extract_vosk_dlls(dest)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(dest))
        except OSError:
            pass
    os.environ["PATH"] = str(dest) + os.pathsep + os.environ.get("PATH", "")


def _extract_vosk_dlls(dest: Path) -> None:
    meta = json.loads(
        urllib.request.urlopen("https://pypi.org/pypi/vosk/" + _VOSK_VER + "/json", timeout=60).read()
    )
    wheel = ""
    for item in meta.get("urls") or []:
        name = str(item.get("filename") or "")
        if name.endswith("win_amd64.whl"):
            wheel = str(item.get("url") or "")
            break
    if not wheel:
        raise RuntimeError("No encontré el wheel de Vosk para Windows.")
    zpath = dest / "vosk.whl"
    urllib.request.urlretrieve(wheel, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name.lower().endswith(".dll"):
                continue
            if "/vosk/" not in "/" + name.lower() and not name.lower().startswith("vosk/"):
                continue
            target = dest / Path(name).name
            target.write_bytes(zf.read(info))
    try:
        zpath.unlink()
    except OSError:
        pass
    if not (dest / "libvosk.dll").is_file():
        raise RuntimeError("No pude extraer libvosk.dll.")


def _vosk_model():
    global _model
    if _model is not None:
        return _model
    _prepare_vosk_native()
    try:
        from vosk import Model, SetLogLevel
    except ImportError as exc:
        raise RuntimeError("Falta Vosk. pip install vosk  (o reconstruí ILSA.exe)") from exc
    except OSError as exc:
        raise RuntimeError(
            "Vosk no cargó las DLL. Dejá que ILSA baje libvosk o reconstruí el exe."
        ) from exc
    SetLogLevel(-1)
    path = _ensure_model_dir()
    print(f"[backend] STT Vosk cargando {path}")
    _model = Model(str(path))
    return _model


def _ensure_model_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ILSA" / "stt"
    dest = base / _MODEL_NAME
    if (dest / "am" / "final.mdl").is_file() or (dest / "conf" / "model.conf").is_file():
        return dest
    base.mkdir(parents=True, exist_ok=True)
    zpath = base / f"{_MODEL_NAME}.zip"
    print(f"[backend] STT bajando modelo español ({_MODEL_NAME})…")
    urllib.request.urlretrieve(_MODEL_URL, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(base)
    try:
        zpath.unlink()
    except OSError:
        pass
    if not dest.is_dir():
        raise RuntimeError("No se pudo extraer el modelo Vosk.")
    return dest


def _wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError("WAV debe ser PCM 16 bit")
    if ch == 1:
        return raw, rate
    out = bytearray()
    step = ch * 2
    for i in range(0, len(raw), step):
        out.extend(raw[i : i + 2])
    return bytes(out), rate


def _resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate:
        return pcm
    import numpy as np

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if x.size < 2:
        return pcm
    n = max(1, int(round(x.size * to_rate / from_rate)))
    idx = np.linspace(0, x.size - 1, n)
    y = np.interp(idx, np.arange(x.size), x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()
