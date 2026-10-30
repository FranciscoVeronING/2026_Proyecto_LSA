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
    "Espera_subtitulos_ms",
    "Espera_voz_ms",
    "Cumple_RNF01",
    "RAM_mb",
    "CPU_percent",
    "Browser_RAM_mb",
    "Browser_CPU_percent",
    "Browser_name",
    "Browser_n_proc",
    "Via",
    "N_glosses",
    "Input_chars",
    "Output_chars",
    "Prompt_tokens",
    "Completion_tokens",
    "Eval_tokens",
    "Sys_tokens",
    "Kv_hit",
    "Kv_mb",
    "T_prompt_eval_ms",
    "T_eval_ms",
    "Tokens_per_s",
    "Modelo",
    "N_ctx",
    "N_tokens",
    "Queue_depth",
    "N_threads",
    "Cloud_RAM_mb",
    "Cloud_CPU_percent",
    "Stt_ms",
    "Tpose_fetch_ms",
    "Tpose_compose_ms",
    "Tpose_gets",
    "Tpose_cache_hits",
    "Tpose_missing_n",
    "Enie_restored",
    "Js_heap_mb",
]

SIGN_CSV_FIELDS = [
    "Timestamp",
    "accepted",
    "added",
    "reason",
    "gloss",
    "conf1",
    "conf2",
    "conf3",
    "conf_margin",
    "classifier_ms",
    "prepare_ms",
    "total_ms",
    "capture_ms",
    "holistic_ms",
    "jpeg_ms",
    "time_to_record_ms",
    "n_frames",
    "n_raw",
    "n_sent",
    "payload_kb",
    "left_handed",
    "RAM_mb",
    "CPU_percent",
    "Browser_RAM_mb",
    "Browser_CPU_percent",
    "Browser_name",
    "Browser_n_proc",
]

_lock = threading.Lock()
_rows: list[dict[str, Any]] = []
_sign_rows: list[dict[str, Any]] = []
_dir_override: Path | None = None


def set_metrics_dir(path: Path | None) -> None:
    global _dir_override, _rows, _sign_rows
    with _lock:
        _dir_override = Path(path) if path is not None else None
        _rows = []
        _sign_rows = []


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


def signs_csv_path() -> Path:
    return metrics_dir() / "metrics_signs.csv"


def _proc_columns(proc: dict[str, Any] | None) -> dict[str, Any]:
    """ILSA vs browser RAM/CPU. RAM_mb/CPU_percent stay the Python process."""
    proc = proc or {}
    ram = proc.get("RAM_mb", proc.get("rss_mb"))
    cpu = proc.get("CPU_percent", proc.get("cpu_percent"))
    return {
        "RAM_mb": ram,
        "CPU_percent": cpu,
        "rss_mb": ram,
        "cpu_percent": cpu,
        "Browser_RAM_mb": proc.get("Browser_RAM_mb"),
        "Browser_CPU_percent": proc.get("Browser_CPU_percent"),
        "Browser_name": proc.get("Browser_name"),
        "Browser_n_proc": proc.get("Browser_n_proc"),
    }


def record_sign_row(fields: dict[str, Any]) -> dict[str, Any]:
    """One TinySkeleton /sign (not a GGUF turn)."""
    from utils.process_stats import snapshot as proc_snap

    proc = {}
    try:
        proc = proc_snap()
    except Exception:
        proc = {}
    row = {
        "Timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        **_proc_columns(proc),
        **{k: v for k, v in fields.items() if v is not None},
    }
    with _lock:
        _sign_rows.append(row)
        dest = signs_csv_path()
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SIGN_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for item in _sign_rows:
                writer.writerow(item)
    return row


def _num(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def transport_ms(roundtrip_ms: Any, queue_ms: Any, inference_ms: Any) -> float:
    """Network RTT with queue and inference stripped out."""
    return round(max(0.0, _num(roundtrip_ms) - _num(queue_ms) - _num(inference_ms)), 1)


def espera_total_ms(
    pause_ms: Any,
    queue_ms: Any = 0,
    inference_ms: Any = 0,
    transport_ms: Any = 0,
    delivery_ms: Any = 0,
) -> float:
    """Sign-end → output: pause + queue + inference + transport + delivery."""
    return round(
        _num(pause_ms)
        + _num(queue_ms)
        + _num(inference_ms)
        + _num(transport_ms)
        + _num(delivery_ms),
        1,
    )


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
    espera_sub = espera_total_ms(pause, queue, infer, transport, delivery)
    ok = meets_rnf01(infer)
    canal = normalize_channel(channel)
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
        "Espera_subtitulos_ms": espera_sub,
        "Espera_voz_ms": None,
        "Cumple_RNF01": ok,
        **_proc_columns(proc),
        "Via": None,
        "N_glosses": None,
        "Input_chars": None,
        "Output_chars": None,
        "Prompt_tokens": None,
        "Completion_tokens": None,
        "Eval_tokens": None,
        "Sys_tokens": None,
        "Kv_hit": None,
        "Kv_mb": None,
        "T_prompt_eval_ms": None,
        "T_eval_ms": None,
        "Tokens_per_s": None,
        "Modelo": None,
        "N_ctx": None,
        "N_tokens": None,
        "Queue_depth": None,
        "N_threads": None,
        "Cloud_RAM_mb": None,
        "Cloud_CPU_percent": None,
        "Stt_ms": None,
        "Tpose_fetch_ms": None,
        "Tpose_compose_ms": None,
        "Tpose_gets": None,
        "Tpose_cache_hits": None,
        "Tpose_missing_n": None,
        "Enie_restored": None,
        "Js_heap_mb": None,
    }
    if extra:
        row.update(extra)
        if extra.get("RAM_mb") is None and extra.get("rss_mb") is not None:
            row["RAM_mb"] = extra.get("rss_mb")
        if extra.get("CPU_percent") is None and extra.get("cpu_percent") is not None:
            row["CPU_percent"] = extra.get("cpu_percent")
    with _lock:
        _rows.append(row)
        _write_locked()
    print(
        f"[metrics] {canal} Tiempo de Pausa (Heurística de Cierre)={pause:.1f} ms "
        f"(NOT model time) queue={queue:.1f} inference={infer:.1f} "
        f"transport={transport:.1f} delivery={delivery:.1f} total={total:.1f} "
        f"espera_sub={row.get('Espera_subtitulos_ms')} espera_voz={row.get('Espera_voz_ms')} "
        f"RNF-01={'ok' if ok else 'fail'} ILSA RAM_mb={row.get('RAM_mb')} "
        f"CPU_percent={row.get('CPU_percent')} "
        f"Browser_RAM_mb={row.get('Browser_RAM_mb')} "
        f"Browser_CPU_percent={row.get('Browser_CPU_percent')}",
        flush=True,
    )
    return row


def _write_locked() -> None:
    dest_json = json_path()
    dest_csv = csv_path()
    dest_json.write_text(
        json.dumps(
            [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in _rows],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with dest_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in _rows:
            out = {k: v for k, v in row.items() if not str(k).startswith("_")}
            out["Cumple_RNF01"] = bool(out.get("Cumple_RNF01"))
            writer.writerow(out)


def rows() -> list[dict[str, Any]]:
    with _lock:
        return list(_rows)


def add_delivery(
    channel: str,
    delivery_ms: float | None = None,
    *,
    espera_subtitulos_ms: float | None = None,
    espera_voz_ms: float | None = None,
) -> dict[str, Any] | None:
    """Fold Meet-side presentation / wait-from-sign-end into the last matching turn."""
    canal = normalize_channel(channel)
    extra = _num(delivery_ms) if delivery_ms is not None else 0.0
    with _lock:
        for row in reversed(_rows):
            if row.get("Canal") != canal:
                continue
            if delivery_ms is not None:
                row["Entrega_ms"] = round(_num(row.get("Entrega_ms")) + extra, 1)
            row["Latencia_Total_ms"] = round(
                _num(row.get("Cola_ms"))
                + _num(row.get("Inferencia_ms"))
                + _num(row.get("Transporte_ms"))
                + _num(row.get("Entrega_ms")),
                1,
            )
            if espera_subtitulos_ms is not None:
                row["Espera_subtitulos_ms"] = round(_num(espera_subtitulos_ms), 1)
                row["_espera_sub_client"] = True
            elif not row.get("_espera_sub_client"):
                row["Espera_subtitulos_ms"] = espera_total_ms(
                    row.get("Pausa_ms"),
                    row.get("Cola_ms"),
                    row.get("Inferencia_ms"),
                    row.get("Transporte_ms"),
                    row.get("Entrega_ms"),
                )
            if espera_voz_ms is not None:
                row["Espera_voz_ms"] = round(_num(espera_voz_ms), 1)
            _write_locked()
            return dict(row)
    return None
