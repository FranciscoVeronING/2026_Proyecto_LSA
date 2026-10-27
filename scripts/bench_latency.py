"""Production semantic latency bench (Cloud Run).

Measures the theoretical stages without mixing channels:

  pause      constant 4 s utterance close (not slept; annotated)
  queue      `queued_ms` reported by the server
  inference  server `ms` (RNF-01: 2 s ceiling on broadband)
  network    roundtrip − queue − inference
  delivery   reverse channel: T-pose download + compositor; forward: 0
             (the Meet HUD does not run in this script)
  total      pause + roundtrip + delivery

`--channel both` runs signer then hearing. It does not fire both POSTs at
once: on Cloud Run `/sign` and `/hearing` share an inference lock, and mixing
them inflates the queue in a way a real ILSA call (one mode) does not.

The signer channel takes real utterances from ``dataset_glosas.json`` (unique,
mixed lengths). It does not invent glosses.

Hearing examples are Spanish because that is the language the model translates.

    python scripts/bench_latency.py --channel signer --n 20
    python scripts/bench_latency.py --channel hearing --n 20
    python scripts/bench_latency.py --channel both --n 15
    python scripts/bench_latency.py --channel hearing --n 10 --no-avatar
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classifier.config import UTTERANCE_PAUSE_SEC
from semantic.remote import last_timing, ping, translate_hearing, translate_sign

PAUSE_MS = round(UTTERANCE_PAUSE_SEC * 1000, 1)

SIGNER_DATASET = Path(
    r"D:\Mis Archivos\Proyectos\2026_Proyecto_semantico_sordo\src\semantic\dataset_glosas.json"
)

HEARING_EXAMPLES = [
    "Hola, buenos días.",
    "Necesito llamar a la policía.",
    "Por favor, ¿me podés ayudar?",
    "Quiero trabajar cerca de casa.",
    "La reunión es hoy a la tarde.",
    "El hospital está cerca.",
    "Gracias por tu ayuda.",
    "¿Cómo estás?",
]


def _gloss_line(item: dict) -> str:
    raw = item.get("glosses") or []
    if isinstance(raw, str):
        tokens = raw.split()
    else:
        tokens = [str(tok).strip() for tok in raw if str(tok).strip()]
    return " ".join(tokens)


def load_signer_examples(path: Path, n: int) -> list[str]:
    """Unique dataset lines, interleaved by length (1…12) so it is not only short phrases."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("dataset") if isinstance(payload, dict) else payload
    by_len: dict[int, list[str]] = {}
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        line = _gloss_line(item)
        if not line or line in seen:
            continue
        seen.add(line)
        by_len.setdefault(len(line.split()), []).append(line)
    if not by_len:
        raise RuntimeError(f"Dataset has no glosses: {path}")
    lengths = sorted(by_len)
    print(
        "Signer dataset: "
        + ", ".join(f"{ln} tok×{len(by_len[ln])}" for ln in lengths),
        flush=True,
    )
    picked: list[str] = []
    idx = {ln: 0 for ln in lengths}
    while len(picked) < n:
        progressed = False
        for ln in lengths:
            bucket = by_len[ln]
            i = idx[ln]
            if i >= len(bucket):
                continue
            picked.append(bucket[i])
            idx[ln] = i + 1
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
    return picked


def _metrics_dir() -> Path:
    import os

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    path = Path(base) / "ILSA" / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def _one_signer(text: str) -> dict:
    t0 = time.perf_counter()
    try:
        spanish = translate_sign(text)
        error = ""
    except Exception as exc:
        spanish = ""
        error = str(exc)
    roundtrip = (time.perf_counter() - t0) * 1000
    timing = last_timing()
    queued = timing.get("queued_ms") or 0.0
    infer = timing.get("infer_ms") if timing.get("infer_ms") is not None else timing.get("server_ms")
    infer = infer if infer is not None else 0.0
    network = max(0.0, roundtrip - queued - infer)
    return {
        "channel": "signer",
        "input": text,
        "n_glosses": len(text.split()),
        "output": spanish,
        "error": error or None,
        "pause_ms": PAUSE_MS,
        "queue_ms": round(queued, 1),
        "inference_ms": round(infer, 1),
        "network_ms": round(network, 1),
        "delivery_ms": 0.0,
        "roundtrip_ms": round(roundtrip, 1),
        "total_with_pause_ms": round(PAUSE_MS + roundtrip, 1),
    }


def _one_hearing(text: str, avatar: bool) -> dict:
    t0 = time.perf_counter()
    try:
        glosses = translate_hearing(text)
        error = ""
    except Exception as exc:
        glosses = ""
        error = str(exc)
    roundtrip = (time.perf_counter() - t0) * 1000
    timing = last_timing()
    queued = timing.get("queued_ms") or 0.0
    infer = timing.get("infer_ms") if timing.get("infer_ms") is not None else timing.get("server_ms")
    infer = infer if infer is not None else 0.0
    network = max(0.0, roundtrip - queued - infer)
    delivery = 0.0
    if avatar and glosses:
        from avatar.remote import clip_for_glosses

        t_del = time.perf_counter()
        try:
            clip_for_glosses(glosses)
        except Exception as exc:
            error = (error + " | " if error else "") + str(exc)
        delivery = (time.perf_counter() - t_del) * 1000
    return {
        "channel": "hearing",
        "input": text,
        "output": glosses,
        "error": error or None,
        "pause_ms": 0.0,
        "queue_ms": round(queued, 1),
        "inference_ms": round(infer, 1),
        "network_ms": round(network, 1),
        "delivery_ms": round(delivery, 1),
        "roundtrip_ms": round(roundtrip, 1),
        "total_with_pause_ms": round(roundtrip + delivery, 1),
    }


def _run_channel(channel: str, pool: list[str], interval: float, avatar: bool) -> list[dict]:
    rows = []
    n = len(pool)
    for i, text in enumerate(pool):
        extra = f" ({len(text.split())} glosses)" if channel == "signer" else ""
        print(f"[{channel} {i + 1}/{n}]{extra} {text!r}", flush=True)
        row = _one_signer(text) if channel == "signer" else _one_hearing(text, avatar)
        row["i"] = i + 1
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rows.append(row)
        print(
            f"  queue={row['queue_ms']:.0f} infer={row['inference_ms']:.0f} "
            f"net={row['network_ms']:.0f} delivery={row['delivery_ms']:.0f} "
            f"rt={row['roundtrip_ms']:.0f} total={row['total_with_pause_ms']:.0f} ms",
            flush=True,
        )
        if i + 1 < n and interval > 0:
            time.sleep(interval)
    return rows


def _summarize(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(row["channel"], []).append(row)
    out: dict = {}
    for channel, items in by.items():
        ok = [r for r in items if not r.get("error")]

        def col(name: str) -> list[float]:
            return [float(r[name]) for r in ok]

        out[channel] = {
            "n": len(items),
            "n_ok": len(ok),
            "queue_p50": _pct(col("queue_ms"), 50),
            "queue_p95": _pct(col("queue_ms"), 95),
            "inference_p50": _pct(col("inference_ms"), 50),
            "inference_p95": _pct(col("inference_ms"), 95),
            "network_p50": _pct(col("network_ms"), 50),
            "delivery_p50": _pct(col("delivery_ms"), 50),
            "delivery_p95": _pct(col("delivery_ms"), 95),
            "roundtrip_p50": _pct(col("roundtrip_ms"), 50),
            "roundtrip_p95": _pct(col("roundtrip_ms"), 95),
            "total_with_pause_p50": _pct(col("total_with_pause_ms"), 50),
            "total_with_pause_p95": _pct(col("total_with_pause_ms"), 95),
            "infer_rnf01_frac_le_2s": (
                round(sum(1 for v in col("inference_ms") if v <= 2000) / len(ok), 3)
                if ok
                else None
            ),
        }
        if ok:
            out[channel]["inference_mean"] = round(statistics.fmean(col("inference_ms")), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue / inference / delivery latencies of the semantic service.")
    parser.add_argument("--channel", choices=("signer", "hearing", "both"), default="both")
    parser.add_argument("--n", type=int, default=15, help="Requests per channel.")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between requests (0 = burst, measures queue).",
    )
    parser.add_argument("--no-avatar", action="store_true", help="Hearing: do not download T-pose.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--signer-dataset",
        type=Path,
        default=SIGNER_DATASET,
        help="JSON {dataset:[{glosses, spanish}, …]} of the forward channel.",
    )
    args = parser.parse_args(argv)

    if not args.no_warmup:
        print("Warmup GET /health …", flush=True)
        if not ping():
            print("Semantic service is not responding. Aborting.", flush=True)
            return 1

    channels = ("signer", "hearing") if args.channel == "both" else (args.channel,)
    signer_pool: list[str] = []
    if "signer" in channels:
        if not args.signer_dataset.is_file():
            print(f"Signer dataset is missing: {args.signer_dataset}", flush=True)
            return 1
        signer_pool = load_signer_examples(args.signer_dataset, args.n)
        print(f"Signer: {len(signer_pool)} valid utterances from the dataset.", flush=True)
    rows: list[dict] = []
    for channel in channels:
        pool = signer_pool if channel == "signer" else [HEARING_EXAMPLES[i % len(HEARING_EXAMPLES)] for i in range(args.n)]
        rows.extend(_run_channel(channel, pool, args.interval, avatar=not args.no_avatar))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = _metrics_dir() / f"bench-{stamp}.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(rows)
    summary_path = dest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSONL  {dest}")
    print(f"Summary {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
