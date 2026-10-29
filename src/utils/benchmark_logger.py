"""Five-segment pipeline timings for RNF-01.

Segments (client + Cloud Run):

  Pause       gesture-idle heuristic (NOT model time)
  Queue       t_inicio_inferencia − t_llegada_servidor  (`queue_ms`)
  Inference   t_fin_inferencia − t_inicio_inferencia    (`inference_ms`, RNF-01)
  Transport   roundtrip − queue − inference             (network only)
  Delivery    client presentation (caption/TTS or T-pose + canvas)

RNF-01: quantized GGUF inference ≤ 1500 ms target, ≤ 2000 ms on broadband.
Pause never counts toward that budget.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHANNEL_DIRECT = "Directo"
CHANNEL_INVERSE = "Inverso"

RNF01_TARGET_MS = 1500.0
RNF01_CEILING_MS = 2000.0

CSV_FIELDS = [
    "Timestamp",
    "Canal",
    "Pausa_ms",
    "Cola_ms",
    "Inferencia_ms",
    "Transporte_ms",
    "Entrega_ms",
    "Latencia_Total_ms",
    "Cumple_RNF01",
]

_lock = threading.Lock()
_rows: list[dict[str, Any]] = []
_dir_override: Path | None = None


def set_metrics_dir(path: Path | None) -> None:
    global _dir_override, _rows
    with _lock:
        _dir_override = Path(path) if path is not None else None
        _rows = []


def metrics_dir() -> Path:
    if _dir_override is not None:
        _dir_override.mkdir(parents=True, exist_ok=True)
        return _dir_override
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_path() -> Path:
    return metrics_dir() / "metrics_breakdown.json"


def csv_path() -> Path:
    return metrics_dir() / "metrics_breakdown.csv"


def _num(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def transport_ms(roundtrip_ms: Any, queue_ms: Any, inference_ms: Any) -> float:
    """Network RTT with queue and inference stripped out."""
    return round(max(0.0, _num(roundtrip_ms) - _num(queue_ms) - _num(inference_ms)), 1)


def meets_rnf01(inference_ms: Any, *, ceiling_ms: float = RNF01_CEILING_MS) -> bool:
    return _num(inference_ms) <= ceiling_ms


def normalize_channel(channel: str) -> str:
    raw = (channel or "").strip().lower()
    if raw in ("inverso", "inverse", "hearing", "oyente"):
        return CHANNEL_INVERSE
    return CHANNEL_DIRECT


def record_turn(
    *,
    channel: str,
    pause_ms: Any = 0,
    queue_ms: Any = 0,
    inference_ms: Any = 0,
    transport_ms: Any = 0,
    delivery_ms: Any = 0,
    timestamp: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pause = round(_num(pause_ms), 1)
    queue = round(_num(queue_ms), 1)
    infer = round(_num(inference_ms), 1)
    transport = round(_num(transport_ms), 1)
    delivery = round(_num(delivery_ms), 1)
    # Pause is the close heuristic; it is not model latency.
    total = round(queue + infer + transport + delivery, 1)
    ok = meets_rnf01(infer)
    canal = normalize_channel(channel)
    proc: dict[str, Any] = {}
    try:
        from utils.process_stats import snapshot as proc_snap

        proc = proc_snap()
    except Exception:
        proc = {}
    row = {
        "Timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "Canal": canal,
        "Pausa_ms": pause,
        "Cola_ms": queue,
        "Inferencia_ms": infer,
        "Transporte_ms": transport,
        "Entrega_ms": delivery,
        "Latencia_Total_ms": total,
        "Cumple_RNF01": ok,
    }
    if extra:
        row.update(extra)
    if proc:
        row.update(proc)
    with _lock:
        _rows.append(row)
        _write_locked()
    print(
        f"[metrics] {canal} Tiempo de Pausa (Heurística de Cierre)={pause:.1f} ms "
        f"(NOT model time) queue={queue:.1f} inference={infer:.1f} "
        f"transport={transport:.1f} delivery={delivery:.1f} total={total:.1f} "
        f"RNF-01={'ok' if ok else 'fail'}",
        flush=True,
    )
    if proc:
        print(
            f"[metrics] process rss_mb={proc.get('rss_mb')} "
            f"cpu_percent={proc.get('cpu_percent')}",
            flush=True,
        )
    return row


def _write_locked() -> None:
    dest_json = json_path()
    dest_csv = csv_path()
    dest_json.write_text(json.dumps(_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with dest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in _rows:
            out = dict(row)
            out["Cumple_RNF01"] = bool(out.get("Cumple_RNF01"))
            writer.writerow(out)


def rows() -> list[dict[str, Any]]:
    with _lock:
        return list(_rows)


def add_delivery(channel: str, delivery_ms: float) -> dict[str, Any] | None:
    """Fold Meet-side presentation into the last matching turn."""
    canal = normalize_channel(channel)
    extra = _num(delivery_ms)
    with _lock:
        for row in reversed(_rows):
            if row.get("Canal") != canal:
                continue
            row["Entrega_ms"] = round(_num(row.get("Entrega_ms")) + extra, 1)
            row["Latencia_Total_ms"] = round(
                _num(row.get("Cola_ms"))
                + _num(row.get("Inferencia_ms"))
                + _num(row.get("Transporte_ms"))
                + _num(row.get("Entrega_ms")),
                1,
            )
            _write_locked()
            return dict(row)
    return None
