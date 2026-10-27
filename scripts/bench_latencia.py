"""Banco de latencia del semántico de producción (Cloud Run).

Mide las etapas del marco teórico, sin mezclar canales:

  pausa      constante 4 s del cierre de enunciado (no se duerme; se anota)
  cola       `queued_ms` que informa el servidor
  inferencia `ms` del servidor (RNF-01: techo 2 s en banda ancha)
  red        roundtrip − cola − inferencia
  entrega    canal inverso: bajada T-pose + compositor; canal directo: 0
             (el HUD de Meet no corre en este script)
  total      pausa + roundtrip + entrega

`--canal ambos` corre sordo y después oyente. No dispara los dos POST a la
vez: en Cloud Run `/sign` y `/hearing` comparten un candado de inferencia, y
mezclarlos infla la cola de un modo que una llamada ILSA (un modo) no produce.

El canal sordo toma enunciados reales de ``dataset_glosas.json`` (únicos,
distintas longitudes). No inventa glosas.

    python scripts/bench_latencia.py --canal sordo --n 20
    python scripts/bench_latencia.py --canal oyente --n 20
    python scripts/bench_latencia.py --canal ambos --n 15
    python scripts/bench_latencia.py --canal oyente --n 10 --sin-avatar
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
from semantic.remote import last_timing, ping, translate_oyente, translate_sordo

PAUSE_MS = round(UTTERANCE_PAUSE_SEC * 1000, 1)

DATASET_SORDO = Path(
    r"D:\Mis Archivos\Proyectos\2026_Proyecto_semantico_sordo\src\semantic\dataset_glosas.json"
)

OYENTE = [
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


def load_sordo_examples(path: Path, n: int) -> list[str]:
    """Únicas del dataset, intercalando longitudes (1…12) para no mandar solo frases cortas."""
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
        raise RuntimeError(f"El dataset no tiene glosas: {path}")
    lengths = sorted(by_len)
    print(
        "Sordo dataset: "
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


def _one_sordo(text: str) -> dict:
    t0 = time.perf_counter()
    try:
        spanish = translate_sordo(text)
        error = ""
    except Exception as exc:
        spanish = ""
        error = str(exc)
    roundtrip = (time.perf_counter() - t0) * 1000
    timing = last_timing()
    queued = timing.get("queued_ms") or 0.0
    infer = timing.get("infer_ms") if timing.get("infer_ms") is not None else timing.get("server_ms")
    infer = infer if infer is not None else 0.0
    red = max(0.0, roundtrip - queued - infer)
    return {
        "canal": "sordo",
        "entrada": text,
        "n_glosas": len(text.split()),
        "salida": spanish,
        "error": error or None,
        "pausa_ms": PAUSE_MS,
        "cola_ms": round(queued, 1),
        "inferencia_ms": round(infer, 1),
        "red_ms": round(red, 1),
        "entrega_ms": 0.0,
        "roundtrip_ms": round(roundtrip, 1),
        "total_con_pausa_ms": round(PAUSE_MS + roundtrip, 1),
    }


def _one_oyente(text: str, avatar: bool) -> dict:
    t0 = time.perf_counter()
    try:
        glosses = translate_oyente(text)
        error = ""
    except Exception as exc:
        glosses = ""
        error = str(exc)
    roundtrip = (time.perf_counter() - t0) * 1000
    timing = last_timing()
    queued = timing.get("queued_ms") or 0.0
    infer = timing.get("infer_ms") if timing.get("infer_ms") is not None else timing.get("server_ms")
    infer = infer if infer is not None else 0.0
    red = max(0.0, roundtrip - queued - infer)
    entrega = 0.0
    if avatar and glosses:
        from avatar.remote import clip_for_glosses

        t_del = time.perf_counter()
        try:
            clip_for_glosses(glosses)
        except Exception as exc:
            error = (error + " | " if error else "") + str(exc)
        entrega = (time.perf_counter() - t_del) * 1000
    return {
        "canal": "oyente",
        "entrada": text,
        "salida": glosses,
        "error": error or None,
        "pausa_ms": 0.0,
        "cola_ms": round(queued, 1),
        "inferencia_ms": round(infer, 1),
        "red_ms": round(red, 1),
        "entrega_ms": round(entrega, 1),
        "roundtrip_ms": round(roundtrip, 1),
        "total_con_pausa_ms": round(roundtrip + entrega, 1),
    }


def _run_canal(canal: str, pool: list[str], intervalo: float, avatar: bool) -> list[dict]:
    rows = []
    n = len(pool)
    for i, text in enumerate(pool):
        extra = f" ({len(text.split())} glosas)" if canal == "sordo" else ""
        print(f"[{canal} {i + 1}/{n}]{extra} {text!r}", flush=True)
        row = _one_sordo(text) if canal == "sordo" else _one_oyente(text, avatar)
        row["i"] = i + 1
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rows.append(row)
        print(
            f"  cola={row['cola_ms']:.0f} infer={row['inferencia_ms']:.0f} "
            f"red={row['red_ms']:.0f} entrega={row['entrega_ms']:.0f} "
            f"rt={row['roundtrip_ms']:.0f} total={row['total_con_pausa_ms']:.0f} ms",
            flush=True,
        )
        if i + 1 < n and intervalo > 0:
            time.sleep(intervalo)
    return rows


def _summarize(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(row["canal"], []).append(row)
    out: dict = {}
    for canal, items in by.items():
        ok = [r for r in items if not r.get("error")]
        def col(name: str) -> list[float]:
            return [float(r[name]) for r in ok]
        out[canal] = {
            "n": len(items),
            "n_ok": len(ok),
            "cola_p50": _pct(col("cola_ms"), 50),
            "cola_p95": _pct(col("cola_ms"), 95),
            "inferencia_p50": _pct(col("inferencia_ms"), 50),
            "inferencia_p95": _pct(col("inferencia_ms"), 95),
            "red_p50": _pct(col("red_ms"), 50),
            "entrega_p50": _pct(col("entrega_ms"), 50),
            "entrega_p95": _pct(col("entrega_ms"), 95),
            "roundtrip_p50": _pct(col("roundtrip_ms"), 50),
            "roundtrip_p95": _pct(col("roundtrip_ms"), 95),
            "total_con_pausa_p50": _pct(col("total_con_pausa_ms"), 50),
            "total_con_pausa_p95": _pct(col("total_con_pausa_ms"), 95),
            "infer_rnf01_frac_le_2s": (
                round(sum(1 for v in col("inferencia_ms") if v <= 2000) / len(ok), 3)
                if ok
                else None
            ),
        }
        if ok:
            out[canal]["inferencia_mean"] = round(statistics.fmean(col("inferencia_ms")), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latencias cola / inferencia / entrega del semántico.")
    parser.add_argument("--canal", choices=("sordo", "oyente", "ambos"), default="ambos")
    parser.add_argument("--n", type=int, default=15, help="Pedidos por canal.")
    parser.add_argument(
        "--intervalo",
        type=float,
        default=2.0,
        help="Segundos entre pedidos (0 = ráfaga, mide cola).",
    )
    parser.add_argument("--sin-avatar", action="store_true", help="Oyente: no bajar T-pose.")
    parser.add_argument("--sin-warmup", action="store_true")
    parser.add_argument(
        "--dataset-sordo",
        type=Path,
        default=DATASET_SORDO,
        help="JSON {dataset:[{glosses, spanish}, …]} del canal directo.",
    )
    args = parser.parse_args(argv)

    if not args.sin_warmup:
        print("Warmup GET /health …", flush=True)
        if not ping():
            print("El semántico no responde. Aborto.", flush=True)
            return 1

    canales = ("sordo", "oyente") if args.canal == "ambos" else (args.canal,)
    sordo_pool: list[str] = []
    if "sordo" in canales:
        if not args.dataset_sordo.is_file():
            print(f"No está el dataset sordo: {args.dataset_sordo}", flush=True)
            return 1
        sordo_pool = load_sordo_examples(args.dataset_sordo, args.n)
        print(f"Sordo: {len(sordo_pool)} enunciados válidos del dataset.", flush=True)
    rows: list[dict] = []
    for canal in canales:
        pool = sordo_pool if canal == "sordo" else [OYENTE[i % len(OYENTE)] for i in range(args.n)]
        rows.extend(_run_canal(canal, pool, args.intervalo, avatar=not args.sin_avatar))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = _metrics_dir() / f"bench-{stamp}.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summarize(rows)
    summary_path = dest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSONL  {dest}")
    print(f"Resumen {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
