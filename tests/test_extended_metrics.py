"""Extended metrics: classifier CSV, KV fields, Ñ restore count."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.metrics import turn_extra  # noqa: E402
from core.enie import restore_enie_report  # noqa: E402
from utils.benchmark_logger import (  # noqa: E402
    CSV_FIELDS,
    SIGN_CSV_FIELDS,
    add_delivery,
    csv_path,
    record_sign_row,
    record_turn,
    set_metrics_dir,
    signs_csv_path,
)
from utils.process_stats import browser_kind  # noqa: E402


def test_turn_extra_maps_kv_fields():
    extra = turn_extra(
        {
            "prompt_tokens": 120,
            "completion_tokens": 8,
            "kv_hit": True,
            "tokens_per_s": 40.2,
            "model_used": "qwen2.5-1.5b.gguf",
            "queue_depth": 1,
            "cloud_rss_mb": 2100.5,
        },
        via="llm",
        n_glosses=4,
        enie_restored=1,
    )
    assert extra["Prompt_tokens"] == 120
    assert extra["Kv_hit"] is True
    assert extra["Modelo"] == "qwen2.5-1.5b.gguf"
    assert extra["Via"] == "llm"
    assert extra["Enie_restored"] == 1


def test_enie_report_counts_restored_tokens():
    text, n = restore_enie_report("YO ANOS 24", spanish="tengo veinticuatro años")
    assert text == "YO AÑOS 24"
    assert n == 1


def test_signs_csv_and_turn_csv(tmp_path: Path):
    set_metrics_dir(tmp_path)
    record_sign_row(
        {
            "accepted": True,
            "added": True,
            "gloss": "HOLA",
            "conf1": 0.91,
            "classifier_ms": 12.4,
            "prepare_ms": 3.1,
            "n_frames": 16,
        }
    )
    record_turn(
        channel="direct",
        pause_ms=4000,
        queue_ms=10,
        inference_ms=900,
        transport_ms=80,
        delivery_ms=20,
        extra={"Via": "llm", "Prompt_tokens": 90, "Kv_hit": True},
    )
    with csv_path().open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == CSV_FIELDS
    assert rows[0]["Via"] == "llm"
    assert rows[0]["Prompt_tokens"] == "90"
    assert rows[0]["Kv_hit"] == "True"
    with signs_csv_path().open(encoding="utf-8", newline="") as fh:
        signs = list(csv.DictReader(fh))
    assert list(signs[0].keys()) == SIGN_CSV_FIELDS
    assert signs[0]["gloss"] == "HOLA"
    assert signs[0]["classifier_ms"] == "12.4"
    assert "Browser_RAM_mb" in CSV_FIELDS
    assert "Espera_subtitulos_ms" in CSV_FIELDS
    assert "Espera_voz_ms" in CSV_FIELDS
    assert rows[0]["Espera_subtitulos_ms"] == "5010.0"
    assert rows[0]["Espera_voz_ms"] == ""


def test_espera_client_overrides_estimate(tmp_path: Path):
    set_metrics_dir(tmp_path)
    record_turn(
        channel="direct",
        pause_ms=4000,
        queue_ms=80,
        inference_ms=900,
        transport_ms=120,
        delivery_ms=0,
    )
    patched = add_delivery(
        "direct",
        15,
        espera_subtitulos_ms=5120,
        espera_voz_ms=5800,
    )
    assert patched is not None
    assert patched["Entrega_ms"] == 15.0
    assert patched["Espera_subtitulos_ms"] == 5120.0
    assert patched["Espera_voz_ms"] == 5800.0
    assert patched["Latencia_Total_ms"] == 80 + 900 + 120 + 15


def test_browser_kind_maps_chrome_and_edge():
    assert browser_kind("chrome.exe") == "chrome"
    assert browser_kind("msedge.exe") == "msedge"
    assert browser_kind("python.exe") is None


if __name__ == "__main__":
    from pathlib import Path as P
    import tempfile

    test_turn_extra_maps_kv_fields()
    print("ok test_turn_extra_maps_kv_fields")
    test_enie_report_counts_restored_tokens()
    print("ok test_enie_report_counts_restored_tokens")
    with tempfile.TemporaryDirectory() as folder:
        test_signs_csv_and_turn_csv(P(folder))
    print("ok test_signs_csv_and_turn_csv")
    with tempfile.TemporaryDirectory() as folder:
        test_espera_client_overrides_estimate(P(folder))
    print("ok test_espera_client_overrides_estimate")
    test_browser_kind_maps_chrome_and_edge()
    print("ok test_browser_kind_maps_chrome_and_edge")
    print("all extended metrics tests passed")
