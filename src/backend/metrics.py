"""Métricas en vivo: JSONL + resumen. Sin etiquetas de verdad; eso es --eval-semantic."""

from __future__ import annotations

import json
import os
import statistics
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_events: deque[dict[str, Any]] = deque(maxlen=400)


def metrics_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    day = datetime.now().strftime("%Y%m%d")
    return metrics_dir() / f"live-{day}.jsonl"


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def log_event(kind: str, **fields: Any) -> dict[str, Any]:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "kind": kind,
        **{k: v for k, v in fields.items() if v is not None},
    }
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        _events.append(record)
        dest = log_path()
        try:
            with dest.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            print(f"[metrics] no pude escribir {dest}: {exc}")
    return record


def _nums(kind: str, key: str) -> list[float]:
    out = []
    for ev in _events:
        if ev.get("kind") != kind:
            continue
        val = ev.get(key)
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


def summary() -> dict[str, Any]:
    with _lock:
        events = list(_events)
    signs = [e for e in events if e.get("kind") == "sign"]
    utts = [e for e in events if e.get("kind") == "utterance"]
    hears = [e for e in events if e.get("kind") == "hearing"]
    accepted = [e for e in signs if e.get("accepted")]
    added = [e for e in accepted if e.get("added")]
    confs = [float(e["conf1"]) for e in accepted if isinstance(e.get("conf1"), (int, float))]
    reasons: dict[str, int] = {}
    for e in signs:
        if not e.get("accepted"):
            r = str(e.get("reason") or "reject")
            reasons[r] = reasons.get(r, 0) + 1
    via = {"literal": 0, "llm": 0, "fallback": 0}
    for e in utts:
        k = str(e.get("via") or "fallback")
        if k not in via:
            k = "fallback"
        via[k] += 1
    clf = _nums("sign", "classifier_ms")
    cap = _nums("sign", "capture_ms")
    hol = _nums("sign", "holistic_ms")
    rem = _nums("utterance", "remote_ms") + _nums("hearing", "remote_ms")
    inf = _nums("utterance", "server_ms") + _nums("hearing", "server_ms")
    return {
        "file": str(log_path()),
        "n_sign": len(signs),
        "n_accepted": len(accepted),
        "n_added": len(added),
        "n_utterance": len(utts),
        "n_hearing": len(hears),
        "reject": reasons,
        "mean_conf1": round(statistics.fmean(confs), 3) if confs else None,
        "via": via,
        "ms": {
            "capture_p50": _pct(cap, 50),
            "capture_p95": _pct(cap, 95),
            "holistic_p50": _pct(hol, 50),
            "classifier_p50": _pct(clf, 50),
            "classifier_p95": _pct(clf, 95),
            "remote_p50": _pct(rem, 50),
            "remote_p95": _pct(rem, 95),
            "llm_server_p50": _pct(inf, 50),
        },
        "last": events[-8:],
    }


def health_line() -> str:
    s = summary()
    ms = s["ms"]
    bits = [f"señas {s['n_accepted']}/{s['n_sign']}"]
    if s["mean_conf1"] is not None:
        bits.append(f"conf {s['mean_conf1']:.0%}")
    if ms.get("classifier_p50") is not None:
        bits.append(f"clf {ms['classifier_p50']:.0f} ms")
    if ms.get("remote_p50") is not None:
        bits.append(f"LLM {ms['remote_p50']:.0f} ms")
    return " · ".join(bits)
