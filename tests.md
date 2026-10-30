# Pipeline performance tests (RNF-01)

This document describes the five-segment latency instrumentation used to
verify **RNF-01**, the local test suite that checks it, and the files that
record each conversational turn.

RNF-01 is an **inference** requirement: the quantized 4-bit GGUF (llama.cpp /
SLM on Cloud Run) must finish a translation in **≤ 1500 ms**, with a **2000 ms**
ceiling on broadband. Gesture-idle pause, HTTP queueing, network RTT, and
client presentation are measured and logged, but they **do not** count toward
that budget.

Clocks: Python uses `time.perf_counter()`; JavaScript uses `performance.now()`.

---

## What was implemented

The pipeline is timed in both directions:

| Channel (CSV `Canal`) | Product mode | Cloud Run | Client output |
|-----------------------|--------------|-----------|---------------|
| **Directo** | Signer (LSA → Spanish) | `POST /sign` (alias `POST /translate`) | Spanish caption on the Meet canvas + Web Speech (`es-AR`) |
| **Inverso** | Hearing (Spanish → LSA) | `POST /hearing` | Gloss list → `.pose` download → skeleton / composite video into Meet |

Each turn is written to `%LOCALAPPDATA%\ILSA\metrics\` as:

- `metrics_breakdown.json`
- `metrics_breakdown.csv`

CSV columns (`metrics_breakdown.csv`): the original five segments plus GGUF /
KV extras (`Prompt_tokens`, `Kv_hit`, `T_eval_ms`, `Modelo`, `Cloud_RAM_mb`,
T-pose, STT, Ñ restores, …). `Latencia_Total_ms` is still Cola + Inferencia +
Transporte + Entrega. **Pausa is stored and labelled, then excluded.**
`Cumple_RNF01` is `true` when `Inferencia_ms ≤ 2000`.

Each TinySkeleton `/sign` is also written to `metrics_signs.csv`
(`classifier_ms`, `prepare_ms`, confidences, capture/Holistic/JPEG, payload).

---

## The five segments

```
  [hands idle]     [HTTP wait]     [GGUF tokens]     [wires only]     [HUD / avatar]
   Tramo 1            Tramo 2          Tramo 3           Tramo 4           Tramo 5
    Pausa              Cola          Inferencia         Transporte         Entrega
     4.0 s            queue_ms       inference_ms      RTT − queue − inf   delivery_ms
   NOT model          server           RNF-01            network             client
```

### 1. Pause — gesture-idle close heuristic

**Not model time.** After a sign, when the hands leave the frame (or stay still),
the capture engine starts `t_pausa_inicio`. If they do not return, the utterance
closes at **4.0 s** (`UTTERANCE_PAUSE_SEC` in `src/classifier/config.py`).

That close is `t_pausa_fin`. Logs use the label:

`Tiempo de Pausa (Heurística de Cierre)` — and state explicitly that it is
**NOT model time**.

| Side | File | What it records |
|------|------|-----------------|
| Extension | `extension/lib/capture.js` | `t_pausa_inicio` when hands leave; `pause_ms` at close |
| ILSA | `src/backend/signer_session.py` | `POST /utterance/end` with `pause_ms`; writes the Directo row |
| Buffer | `src/core/utterance.py` | Same 4 s idle rule on the engine |

The inverse channel has no gesture pause (`Pausa_ms = 0`): speech / chat already
delivers a finished sentence.

### 2. Queue — Cloud Run worker wait

On every `POST /sign`, `/translate`, and `/hearing`, FastAPI stamps
`t_llegada_servidor` when the request hits `infer_slot()`. Both channels share
one llama.cpp mutex, so a second request waits.

`t_inicio_inferencia` is the instant the worker acquires the lock.

```
queue_ms = (t_inicio_inferencia − t_llegada_servidor) × 1000
```

Returned in JSON as `queue_ms` (and the legacy alias `queued_ms`).

Source: sibling repo `2026_Proyecto_LSA` (`src/semantic/service.py`,
`src/semantic/api.py`). Until that service is redeployed, the client still
reads `queued_ms` / `ms` as fallbacks.

### 3. Inference — GGUF SLM (RNF-01)

`t_fin_inferencia` is when the last token / translated text is produced.

```
inference_ms = (t_fin_inferencia − t_inicio_inferencia) × 1000
```

Returned as `inference_ms` (legacy `ms` keeps the same value). This is the only
field that sets `Cumple_RNF01`.

The local suite asserts simulated inference **≤ 1500 ms**. Live Cloud Run is
judged against the **2000 ms** broadband ceiling.

### 4. Transport — network RTT only

The ILSA HTTP client stamps `t_envio` before `POST` and `t_recepcion` after the
full body is read (`src/semantic/remote.py`).

```
roundtrip_ms  = (t_recepcion − t_envio) × 1000
transport_ms  = roundtrip_ms − queue_ms − inference_ms
```

Queue and inference are subtracted so the segment is **wire time only** and
does not overlap the server. Negative results are clamped to 0.

### 5. Delivery — presentation on the client

`t_inicio_entrega` starts when the translated payload is in hand.

| Channel | Measured until |
|---------|----------------|
| Directo | Spanish is drawn on the Meet canvas **and** Web Speech has been started (`extension/content/meet.js`) |
| Inverso | ILSA downloads `.pose` files and composes the clip (`hearing_session.py` + `avatar.remote.clip_for_glosses`), then Meet draws the skeleton and publishes the composite (`meet.js`) |

Meet reports extra presentation time with `POST /metrics/client`, which folds
into the last matching row via `add_delivery()`.

T-pose GETs (`tpose-server`) add `X-ILSA-Ms`; that time sits inside inverse
delivery, not inside GGUF inference.

---

## How a live turn is recorded

**Directo (signer)**

1. Holistic WASM in the offscreen sandbox yields landmarks.
2. `capture.js` crops a sign; ILSA classifies it (`POST /sign` on localhost).
3. Gloss tokens sit in `UtteranceBuffer` until 4 s of idle.
4. Extension `POST /utterance/end` `{ pause_ms }` → ILSA `POST` Cloud Run `/sign`.
5. JSON `queue_ms` + `inference_ms` + client `transport_ms` are stored; pause is
   logged as heuristic, not model time.
6. Meet paints the caption, speaks Spanish, then `POST /metrics/client`
   `{ channel: "direct", delivery_ms }`.

**Inverso (hearing)**

1. Vosk / Meet chat yields Spanish.
2. A single word skips Cloud Run (gloss = that word). A sentence uses
   `POST /hearing`.
3. ILSA times T-pose download + compose as the first delivery slice.
4. Meet applies the avatar clip and reports a second `delivery_ms`.

FPS (Holistic sandbox and `canvas.captureStream(24)`) and ILSA RAM/CPU
(`src/utils/process_stats.py`) are logged on the same `/metrics/client` and
`/health` paths. They are extra columns in JSON; they are not CSV fields.

Collector: `src/utils/benchmark_logger.py`. Session JSONL (older live log)
stays in `src/backend/metrics.py`.

---

## Test suite: `tests/test_performance_pipeline.py`

This file **does not call Cloud Run**. It simulates the server with a local
lock (`fake_slm`) so queue vs inference vs transport can be asserted on a
developer machine.

| Helper | Role |
|--------|------|
| `fake_slm(infer_s)` | `t_llegada` → mutex → sleep `infer_s` → `queue_ms` + `inference_ms` |
| `client_roundtrip(infer_s, extra_net_s)` | Client RTT around the fake SLM; optional extra sleep as network |

| Test | What it proves |
|------|----------------|
| `test_transport_excludes_queue_and_inference` | A second thread waits on the lock (`queue_ms ≥ 80`). Extra network sleep appears in `transport_ms`, not in queue or inference. `roundtrip ≈ queue + inference + transport`. |
| `test_rnf01_inference_budget` | Simulated `inference_ms ≤ 1500`. `meets_rnf01` is true at 1500 and 2000, false at 2001. |
| `test_pause_is_not_model_time` | Pause is stored (`Pausa_ms`) but **not** added into `Latencia_Total_ms`. |
| `test_export_json_csv_both_channels` | One Directo row and one Inverso row; CSV header matches `CSV_FIELDS`; inverse total = 40+1100+90+310. |
| `test_simulated_direct_and_inverse_pipeline` | End-to-end fake Directo (4 s pause annotated) and Inverso (pause 0) both stay under the 1500 ms inference target. |

Output of a `__main__` run goes to `tests/_perf_out/` (gitignored).

### How to run

From the repo root, with `src` on `PYTHONPATH` (the script inserts it itself):

```bash
python tests/test_performance_pipeline.py
```

Or with pytest (each test takes a `tmp_path`):

```bash
pytest tests/test_performance_pipeline.py
```

Expected console: five `ok …` lines and paths to the JSON/CSV written under
`tests/_perf_out/` (script mode) or pytest’s temp dir.

---

## Live Cloud Run bench (optional)

`scripts/bench_latency.py` hits the **real** semantic service. It does not
replace the unit suite: it measures production queue / inference / network
(and inverse T-pose delivery). Pause is annotated as 4000 ms and not slept.

```bash
python scripts/bench_latency.py --channel signer --n 20
python scripts/bench_latency.py --channel hearing --n 20
python scripts/bench_latency.py --channel both --n 15
```

`--channel both` runs signer **then** hearing. It does not fire both POSTs at
once: `/sign` and `/hearing` share the inference lock, and overlapping them
would inflate queue in a way a real one-mode ILSA call does not.

Meet HUD delivery is not exercised by this script.

---

## Key files

| Path | Role |
|------|------|
| `tests/test_performance_pipeline.py` | Offline five-segment + RNF-01 assertions |
| `src/utils/benchmark_logger.py` | Turn collector, transport formula, JSON/CSV export |
| `src/utils/process_stats.py` | Client RSS / CPU |
| `src/semantic/remote.py` | `t_envio` / `t_recepcion`, `transport_ms` |
| `src/backend/signer_session.py` | Directo close + pause log |
| `src/backend/hearing_session.py` | Inverso queue/inference/transport + T-pose delivery |
| `src/backend/server.py` | `POST /utterance/end`, `POST /metrics/client` |
| `extension/lib/capture.js` | Pause start/end |
| `extension/content/meet.js` | Direct caption+TTS and inverse present time |
| `extension/content/camera-hook.js` | `captureStream` FPS |
| `extension/offscreen.js` | Holistic FPS |
| `scripts/bench_latency.py` | Production Cloud Run bench |

Cloud Run (not in this repo): `2026_Proyecto_LSA` — `queue_ms` / `inference_ms`
plus KV/token stats (`kv_hit`, `prompt_tokens`, `t_prompt_eval_ms`, `t_eval_ms`,
`queue_depth`, `cloud_rss_mb`) on `/sign`, `/translate`, `/hearing`.
T-pose: `2026_Proyecto_avatar` — `X-ILSA-Ms`.

KV cache metrics apply to the **GGUF translator**, not TinySkeleton. Classifier
time is `classifier_ms` in `live-*.jsonl` and `metrics_signs.csv`.
