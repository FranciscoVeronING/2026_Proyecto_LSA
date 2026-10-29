"""RAM / CPU of the ILSA process. Sampled in the background so CPU is not always 0."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

_lock = threading.Lock()
_last: dict[str, Any] = {"rss_mb": None, "cpu_percent": None}
_last_ts = 0.0
_proc = None
_sampler_started = False


def _psutil_process():
    global _proc
    import psutil

    if _proc is None or _proc.pid != os.getpid():
        _proc = psutil.Process(os.getpid())
        _proc.cpu_percent(interval=None)
    return _proc


def rss_mb() -> float | None:
    try:
        return round(_psutil_process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return round(counters.WorkingSetSize / (1024 * 1024), 1)
        except Exception:
            pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024 * 1024), 1)
        return round(usage / 1024, 1)
    except Exception:
        return None


def cpu_percent() -> float | None:
    """Process CPU since the previous sample (0–100+ if multiple cores)."""
    try:
        return round(_psutil_process().cpu_percent(interval=None), 1)
    except Exception:
        return None


def _refresh_locked() -> dict[str, Any]:
    global _last_ts
    ram = rss_mb()
    cpu = cpu_percent()
    if ram is not None:
        _last["rss_mb"] = ram
        _last["RAM_mb"] = ram
    if cpu is not None:
        _last["cpu_percent"] = cpu
        _last["CPU_percent"] = cpu
    _last_ts = time.monotonic()
    return dict(_last)


def snapshot() -> dict[str, Any]:
    with _lock:
        if _last.get("rss_mb") is not None and (time.monotonic() - _last_ts) < 0.4:
            return dict(_last)
        return _refresh_locked()


def _sampler_loop() -> None:
    while True:
        time.sleep(1.0)
        try:
            with _lock:
                _refresh_locked()
        except Exception:
            pass


def start_sampler() -> None:
    """Prime CPU accounting and sample once per second."""
    global _sampler_started
    with _lock:
        if _sampler_started:
            return
        _sampler_started = True
        _refresh_locked()
    thread = threading.Thread(target=_sampler_loop, name="ilsa-proc-stats", daemon=True)
    thread.start()
    print(
        f"[metrics] process sampler on pid={os.getpid()} "
        f"rss_mb={_last.get('rss_mb')} cpu_percent={_last.get('cpu_percent')}",
        flush=True,
    )
