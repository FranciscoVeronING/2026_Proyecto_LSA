"""Load and run both channels. One lock serializes inference."""

from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from semantic.config import BACKUP_GGUF, HEARING_DIR, PRIMARY_GGUF, SIGN_DIR
from semantic.engine import LoadedModel, load_gguf
from semantic.gguf import disk_inventory, list_ggufs
from semantic.literals import collapse_letter_digit_runs

_INFER_LOCK = threading.RLock()
_LOAD_LOCK = threading.Lock()
_LOADED = False
_LOADING = False
_LOAD_ERROR: Optional[str] = None

_DEFAULT_SIGN_PROMPT = (
    "Sos un traductor estricto de Lengua de Señas Argentina (LSA) "
    "a español rioplatense natural."
)
_DEFAULT_HEARING_PROMPT = (
    "Traducí español a glosas LSA en mayúsculas, separadas por espacios. "
    "Solo la línea de glosas."
)


def _read_sys_prompt(folder: Path, fallback: str) -> str:
    for name in ("sys_prompt.txt", "system_prompt.txt"):
        path = folder / name
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                print(f"[semantic] Prompt: {path}", flush=True)
                return text
    return fallback


def _n_threads() -> int:
    return max(1, os.cpu_count() or 2)


@dataclass
class Channel:
    name: str
    directory: Path
    user_prefix: str
    collapse_literals: bool = False
    system_prompt: str = ""
    models: list[LoadedModel] = field(default_factory=list)
    active: int = 0

    @property
    def current(self) -> Optional[LoadedModel]:
        if not self.models:
            return None
        return self.models[self.active]

    def load_primary(self, n_threads: int) -> None:
        paths = list_ggufs(self.directory)
        if not paths:
            raise FileNotFoundError(
                f"[semantic] Missing {PRIMARY_GGUF} and {BACKUP_GGUF} in {self.directory}."
            )
        loaded: Optional[LoadedModel] = None
        for path in paths:
            loaded = load_gguf(path, n_threads, self.name)
            if loaded is not None:
                print(f"[semantic] Active {self.name}: {loaded.path}", flush=True)
                break
        if loaded is None:
            names = ", ".join(p.name for p in paths)
            raise RuntimeError(f"[semantic] No GGUF loaded for {self.name} ({names}).")
        self.models = [loaded]
        self.active = 0
        loaded.pin_system(self.system_prompt)

    def try_load_backup(self) -> None:
        have = {m.path.resolve() for m in self.models}
        n_threads = _n_threads()
        for path in list_ggufs(self.directory):
            if path.resolve() in have:
                continue
            nxt = load_gguf(path, n_threads, self.name)
            if nxt is None:
                continue
            nxt.pin_system(self.system_prompt)
            self.models.append(nxt)
            return

    def translate(self, user_text: str) -> tuple[str, str]:
        raw = user_text.strip()
        if self.collapse_literals:
            tokens = [t for t in raw.split() if t]
            collapsed = " ".join(collapse_letter_digit_runs(tokens))
            if collapsed != raw:
                print(
                    f"[semantic] Collapsed letter/digit run: {raw!r} → {collapsed!r}",
                    flush=True,
                )
            raw = collapsed
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{self.user_prefix} {raw}"},
        ]
        last_err = None
        start = self.active
        step = 0
        while step < len(self.models):
            idx = (start + step) % len(self.models)
            model = self.models[idx]
            step += 1
            try:
                text = model.complete(messages, raw, self.name)
                self.active = idx
                return text, model.path.name
            except Exception as exc:
                last_err = exc
                print(f"[semantic] Fail {self.name} {model.path.name}: {exc}", flush=True)
                if model.path.name.lower() == PRIMARY_GGUF:
                    print(f"[semantic] Switching {self.name} to {BACKUP_GGUF}", flush=True)
                    self.try_load_backup()
        raise RuntimeError(f"[semantic] /{self.name} has no usable model: {last_err}")

    def close(self) -> None:
        for model in self.models:
            model.close()
        self.models.clear()
        self.active = 0


SIGN = Channel("sign", SIGN_DIR, "Glosas:", collapse_literals=True)
HEARING = Channel("hearing", HEARING_DIR, "Español:", collapse_literals=False)


@contextmanager
def infer_slot() -> Iterator[float]:
    """Serialize llama.cpp. Yielded seconds are queue wait (t_inicio_inferencia − t_llegada)."""
    t_llegada = time.perf_counter()
    got = _INFER_LOCK.acquire(blocking=False)
    if not got:
        print("[semantic] Queue (server): model busy, waiting…", flush=True)
        _INFER_LOCK.acquire()
    waited = time.perf_counter() - t_llegada
    print(
        f"[semantic] Queue (server): t_cola={waited * 1000:.1f} ms "
        f"(t_llegada_servidor → t_inicio_inferencia)",
        flush=True,
    )
    try:
        yield waited
    finally:
        _INFER_LOCK.release()


def load_status() -> dict:
    return {"loading": _LOADING, "error": _LOAD_ERROR}


def health_snapshot() -> dict:
    sign = SIGN.current
    hearing = HEARING.current
    status = load_status()
    return {
        "ok": True,
        "ready": sign is not None and hearing is not None,
        "loading": status["loading"],
        "load_error": status["error"],
        "primary": PRIMARY_GGUF,
        "backup": BACKUP_GGUF,
        "sign": disk_inventory(SIGN_DIR),
        "hearing": disk_inventory(HEARING_DIR),
        "sign_gguf": str(sign.path) if sign else None,
        "hearing_gguf": str(hearing.path) if hearing else None,
        "model_used_sign": sign.path.name if sign else None,
        "model_used_hearing": hearing.path.name if hearing else None,
        **_process_stats(),
    }


def _process_stats() -> dict:
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        return {
            "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "cpu_percent": proc.cpu_percent(interval=None),
        }
    except Exception:
        return {}


def unload_models() -> None:
    global _LOADED
    SIGN.close()
    HEARING.close()
    SIGN.system_prompt = ""
    HEARING.system_prompt = ""
    _LOADED = False
    gc.collect()


def load_models(force: bool = False) -> None:
    global _LOADED, _LOADING, _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED and SIGN.current and HEARING.current and not force:
            return
        if _LOADED:
            unload_models()
        _LOADING = True
        _LOAD_ERROR = None
        try:
            n_threads = _n_threads()
            SIGN.system_prompt = _read_sys_prompt(SIGN_DIR, _DEFAULT_SIGN_PROMPT)
            HEARING.system_prompt = _read_sys_prompt(HEARING_DIR, _DEFAULT_HEARING_PROMPT)
            print("[semantic] Warmup: local GGUF per channel.", flush=True)
            for ch in (SIGN, HEARING):
                names = []
                if ch.directory.is_dir():
                    names = sorted(p.name for p in ch.directory.iterdir())
                print(
                    f"[semantic] Dir {ch.name} {ch.directory}: {names or '(empty or missing)'}",
                    flush=True,
                )
            SIGN.load_primary(n_threads)
            HEARING.load_primary(n_threads)
            _LOADED = True
            print(f"[semantic] Loaded sign → {SIGN.current.path if SIGN.current else '—'}", flush=True)
            print(
                f"[semantic] Loaded hearing → {HEARING.current.path if HEARING.current else '—'}",
                flush=True,
            )
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            raise
        finally:
            _LOADING = False


def translate_sign(glosses: str) -> tuple[str, str]:
    if not _LOADED or SIGN.current is None:
        load_models()
    return SIGN.translate(glosses)


def translate_hearing(spanish: str) -> tuple[str, str]:
    if not _LOADED or HEARING.current is None:
        load_models()
    return HEARING.translate(spanish)
