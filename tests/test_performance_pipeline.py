"""Simulate the five pipeline segments and the RNF-01 inference budget.

Does not call Cloud Run. A local fake SLM records queue + inference with
``time.perf_counter``; the client formula strips them from the RTT so transport
is network-only. Pause is stored and must not enter the model budget.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.benchmark_logger import (  # noqa: E402
    CHANNEL_DIRECT,
    CHANNEL_INVERSE,
    CSV_FIELDS,
    RNF01_CEILING_MS,
    RNF01_TARGET_MS,
    csv_path,
    json_path,
    meets_rnf01,
    record_turn,
    set_metrics_dir,
    transport_ms,
)

INFER_LOCK = threading.Lock()


def fake_slm(infer_s: float, queue_hold: threading.Event | None = None) -> dict:
    """Stand-in for Cloud Run ``POST /sign`` / ``POST /hearing``."""
    t_llegada = time.perf_counter()
    INFER_LOCK.acquire()
    try:
        t_inicio = time.perf_counter()
        queue_ms = (t_inicio - t_llegada) * 1000
        if queue_hold is not None:
            queue_hold.wait(timeout=1.0)
        time.sleep(infer_s)
        t_fin = time.perf_counter()
        inference_ms = (t_fin - t_inicio) * 1000
    finally:
        INFER_LOCK.release()
    return {
        "queue_ms": round(queue_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "queued_ms": round(queue_ms, 1),
        "ms": round(inference_ms, 1),
    }


def client_roundtrip(infer_s: float, extra_net_s: float = 0.0) -> dict:
    t_envio = time.perf_counter()
    payload = fake_slm(infer_s)
    if extra_net_s:
        time.sleep(extra_net_s)
    t_recepcion = time.perf_counter()
    roundtrip = (t_recepcion - t_envio) * 1000
    net = transport_ms(roundtrip, payload["queue_ms"], payload["inference_ms"])
    payload["roundtrip_ms"] = round(roundtrip, 1)
    payload["transport_ms"] = net
    return payload


def test_transport_excludes_queue_and_inference(tmp_path: Path) -> None:
    set_metrics_dir(tmp_path)
    queued: dict = {}

    def hold() -> None:
        fake_slm(0.15)

    def waiter() -> None:
        queued.update(fake_slm(0.04))

    first = threading.Thread(target=hold)
    second = threading.Thread(target=waiter)
    first.start()
    time.sleep(0.04)
    second.start()
    first.join()
    second.join()
    assert queued["queue_ms"] >= 80
    timed = client_roundtrip(0.05, extra_net_s=0.04)
    assert timed["transport_ms"] >= 30
    overlap = timed["queue_ms"] + timed["inference_ms"]
    assert abs(timed["roundtrip_ms"] - (overlap + timed["transport_ms"])) <= 15


def test_rnf01_inference_budget(tmp_path: Path) -> None:
    set_metrics_dir(tmp_path)
    payload = client_roundtrip(0.05)
    assert payload["inference_ms"] <= RNF01_TARGET_MS
    assert meets_rnf01(payload["inference_ms"])
    assert meets_rnf01(RNF01_TARGET_MS)
    assert meets_rnf01(RNF01_CEILING_MS)
    assert not meets_rnf01(RNF01_CEILING_MS + 1)


def test_pause_is_not_model_time(tmp_path: Path) -> None:
    set_metrics_dir(tmp_path)
    t_pausa_inicio = time.perf_counter()
    time.sleep(0.05)
    pause_ms = (time.perf_counter() - t_pausa_inicio) * 1000
    payload = client_roundtrip(0.04)
    row = record_turn(
        channel="direct",
        pause_ms=pause_ms,
        queue_ms=payload["queue_ms"],
        inference_ms=payload["inference_ms"],
        transport_ms=payload["transport_ms"],
        delivery_ms=12.5,
    )
    assert row["Canal"] == CHANNEL_DIRECT
    assert row["Pausa_ms"] >= 40
    model_ms = row["Cola_ms"] + row["Inferencia_ms"] + row["Transporte_ms"] + row["Entrega_ms"]
    assert row["Latencia_Total_ms"] == round(model_ms, 1)
    assert row["Pausa_ms"] not in (row["Cola_ms"], row["Inferencia_ms"], row["Transporte_ms"])
    assert row["Latencia_Total_ms"] < row["Pausa_ms"] + model_ms


def test_export_json_csv_both_channels(tmp_path: Path) -> None:
    set_metrics_dir(tmp_path)
    record_turn(
        channel="direct",
        pause_ms=4000,
        queue_ms=80,
        inference_ms=900,
        transport_ms=120,
        delivery_ms=25,
    )
    record_turn(
        channel="inverse",
        pause_ms=0,
        queue_ms=40,
        inference_ms=1100,
        transport_ms=90,
        delivery_ms=310,
    )
    data = json.loads(json_path().read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["Canal"] == CHANNEL_DIRECT
    assert data[1]["Canal"] == CHANNEL_INVERSE
    assert data[0]["Cumple_RNF01"] is True
    with csv_path().open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == CSV_FIELDS
    assert "RAM_mb" in rows[0]
    assert "CPU_percent" in rows[0]
    assert rows[1]["Canal"] == CHANNEL_INVERSE
    total_inv = float(rows[1]["Latencia_Total_ms"])
    assert total_inv == 40 + 1100 + 90 + 310
    assert data[0].get("RAM_mb") is not None or data[0].get("rss_mb") is not None


def test_simulated_direct_and_inverse_pipeline(tmp_path: Path) -> None:
    set_metrics_dir(tmp_path)
    direct = client_roundtrip(0.06)
    assert direct["inference_ms"] <= RNF01_TARGET_MS
    record_turn(
        channel="Directo",
        pause_ms=4000,
        queue_ms=direct["queue_ms"],
        inference_ms=direct["inference_ms"],
        transport_ms=direct["transport_ms"],
        delivery_ms=8,
    )
    inverse = client_roundtrip(0.07)
    assert inverse["inference_ms"] <= RNF01_TARGET_MS
    record_turn(
        channel="Inverso",
        pause_ms=0,
        queue_ms=inverse["queue_ms"],
        inference_ms=inverse["inference_ms"],
        transport_ms=inverse["transport_ms"],
        delivery_ms=40,
    )
    payload = json.loads(json_path().read_text(encoding="utf-8"))
    assert {row["Canal"] for row in payload} == {CHANNEL_DIRECT, CHANNEL_INVERSE}


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent / "_perf_out"
    dest.mkdir(exist_ok=True)
    tests = [
        test_transport_excludes_queue_and_inference,
        test_rnf01_inference_budget,
        test_pause_is_not_model_time,
        test_export_json_csv_both_channels,
        test_simulated_direct_and_inverse_pipeline,
    ]
    for fn in tests:
        fn(dest)
        print(f"ok {fn.__name__}")
    print(f"wrote {json_path()} {csv_path()}")
