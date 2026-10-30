"""RAM / CPU of ILSA and of the browser, sampled separately.

ILSA is this Python process (`os.getpid()`). The browser is the Chrome or Edge
process *family* (all `chrome.exe` / `msedge.exe` children), which is what
Task Manager shows when those rows are grouped. JS heap from the extension is
a different number and is logged elsewhere.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

_lock = threading.Lock()
_last: dict[str, Any] = {
    "rss_mb": None,
    "cpu_percent": None,
    "RAM_mb": None,
    "CPU_percent": None,
    "Ilsa_RAM_mb": None,
    "Ilsa_CPU_percent": None,
    "Browser_RAM_mb": None,
    "Browser_CPU_percent": None,
    "Browser_name": None,
    "Browser_n_proc": None,
}
_last_ts = 0.0
_proc = None
_sampler_started = False
_browser_handles: dict[int, Any] = {}

# Windows image names plus POSIX / macOS process names.
_BROWSER_EXES: dict[str, str] = {
    "chrome.exe": "chrome",
    "chrome": "chrome",
    "google chrome": "chrome",
    "chromium.exe": "chrome",
    "chromium": "chrome",
    "msedge.exe": "msedge",
    "msedge": "msedge",
    "microsoftedge.exe": "msedge",
    "microsoft edge": "msedge",
}


def browser_kind(process_name: str | None) -> str | None:
    """Map a process image name to chrome / msedge, or None."""
    if not process_name:
        return None
    return _BROWSER_EXES.get(process_name.strip().lower())


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

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_PMC),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
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
    """ILSA process CPU since the previous sample (0–100+ if multiple cores)."""
    try:
        return round(_psutil_process().cpu_percent(interval=None), 1)
    except Exception:
        return None


def _sample_browser_locked() -> None:
    """Sum RSS/CPU of each browser family; keep the family with most RAM."""
    try:
        import psutil
    except Exception:
        return

    buckets: dict[str, dict[str, float]] = {}
    live: set[int] = set()
    self_pid = os.getpid()
    try:
        listed = list(psutil.process_iter(["pid", "name"]))
    except Exception:
        listed = []

    for info in listed:
        try:
            pid = int(info.info.get("pid") or 0)
            kind = browser_kind(info.info.get("name"))
        except Exception:
            continue
        if not kind or pid <= 0 or pid == self_pid:
            continue
        live.add(pid)
        handle = _browser_handles.get(pid)
        if handle is None:
            try:
                handle = psutil.Process(pid)
                handle.cpu_percent(interval=None)
                _browser_handles[pid] = handle
                cpu = 0.0
            except (psutil.Error, OSError):
                continue
        else:
            try:
                cpu = float(handle.cpu_percent(interval=None) or 0.0)
            except (psutil.Error, OSError):
                _browser_handles.pop(pid, None)
                continue
        try:
            rss = float(handle.memory_info().rss)
        except (psutil.Error, OSError):
            _browser_handles.pop(pid, None)
            continue
        bucket = buckets.setdefault(kind, {"rss": 0.0, "cpu": 0.0, "n": 0.0})
        bucket["rss"] += rss
        bucket["cpu"] += cpu
        bucket["n"] += 1.0

    for pid in list(_browser_handles):
        if pid not in live:
            _browser_handles.pop(pid, None)

    if not buckets:
        _last["Browser_RAM_mb"] = None
        _last["Browser_CPU_percent"] = None
        _last["Browser_name"] = None
        _last["Browser_n_proc"] = None
        return

    kind = max(buckets, key=lambda name: buckets[name]["rss"])
    chosen = buckets[kind]
    _last["Browser_RAM_mb"] = round(chosen["rss"] / (1024 * 1024), 1)
    _last["Browser_CPU_percent"] = round(chosen["cpu"], 1)
    _last["Browser_name"] = kind
    _last["Browser_n_proc"] = int(chosen["n"])


def _refresh_locked() -> dict[str, Any]:
    global _last_ts
    ram = rss_mb()
    cpu = cpu_percent()
    if ram is not None:
        _last["rss_mb"] = ram
        _last["RAM_mb"] = ram
        _last["Ilsa_RAM_mb"] = ram
    if cpu is not None:
        _last["cpu_percent"] = cpu
        _last["CPU_percent"] = cpu
        _last["Ilsa_CPU_percent"] = cpu
    _sample_browser_locked()
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
    browser = _last.get("Browser_name")
    browser_bit = (
        f" browser={browser} rss_mb={_last.get('Browser_RAM_mb')} "
        f"cpu_percent={_last.get('Browser_CPU_percent')} n={_last.get('Browser_n_proc')}"
        if browser
        else " browser=none"
    )
    print(
        f"[metrics] process sampler on pid={os.getpid()} "
        f"ilsa rss_mb={_last.get('rss_mb')} cpu_percent={_last.get('cpu_percent')}"
        f"{browser_bit}",
        flush=True,
    )
