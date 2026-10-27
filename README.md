# ILSA — `extension` branch

Argentine Sign Language (LSA) interpreter for Google Meet: a Chrome extension
plus a local engine. It translates **LSA ↔ Spanish** in the call. Video never
leaves the PC. Cloud Run only sees glosses or Spanish, never JPEG.

This branch is the **client**. The GGUF lives on Cloud Run (`semantic-cloud`).
`.pose` clips live on Cloud Run (`avatar-cloud` / `tpose-server`).

```
Chrome (extension)
  Meet: camera, mic, chat, Holistic WASM, canvas, HUD
        │ HTTP 127.0.0.1:8765  (the service worker proxies lsa-api;
        │                       Meet does not fetch loopback)
        ▼
ILSA (window / ILSA.exe)
  Signer:  TinySkeleton CPU → 4 s pause → POST /sign
  Hearing: Vosk and/or chat → POST /hearing → T-pose + compositor
        │                              │
        ▼                              ▼
  ilsa-semantic-service          tpose-server
  us-central1                    southamerica-east1
  POST /sign  POST /hearing      GET /tposes/{STEM}.pose
```

With ILSA on, Meet does **not** send the raw camera stream: it sends
`canvas.captureStream`. The physical camera is opened **once** (Windows often
takes the device in exclusive mode; a second `getUserMedia` fails with
`NotReadableError`).

---

## Modes

Pick the mode **before** Start. It cannot change mid-session (Restart).

| Mode | Who | Input | Output in Meet |
|------|-----|-------|----------------|
| **Signer** | The person who signs | Camera → Holistic → `POST /sign` | Spanish under the video (optional local TTS, not mixed into the call) |
| **Hearing** | The person who speaks / types | Meet mic and/or chat | Gloss overlay + skeleton on the right |

In signer mode, **Left-handed / Right-handed** (Right-handed by default). The
classifier was trained right-handed: left-handed mirrors X and swaps hands.

**Sign (signer).** Starts with one usable hand. Ends at 60 frames or when both
hands are missing ~0.4 s. Cropped to 16×225 (pose+hands; no face). Low
confidence or repeats stay out of the utterance. After **4 s** without signing
it closes and goes to the translator.

**Hearing.** The audio tap does **not** call `getUserMedia` again: it copies
Meet's track. A single word skips Cloud Run (goes straight to T-pose). A
sentence uses `POST /hearing`.

---

## Layout

```
run_backend.py                 Entry (window or --headless)
LSABackend.bat                 Windows shortcut
requirements.txt               Engine pip deps (no MediaPipe)
assets/                        Logo
packaging/                     PyInstaller, Holistic WASM, icons, zip
scripts/bench_latency.py       Queue / inference / delivery bench
src/
  backend/                     FastAPI 8765 + Tk window
    window.py                  Splash (semantic + T-pose ping), mode, hand
    server.py                  Routes
    signer_session.py          Classifier + buffer + remote /sign
    hearing_session.py         Incoming Vosk; /hearing + clips
    metrics.py                 JSONL under %LOCALAPPDATA%\ILSA\metrics
  classifier/                  TinySkeleton + weights/
  core/                        Landmarks, utterance, repeat policy
  semantic/                    Cloud Run HTTP client
  avatar/                      T-pose client (stems, GET, cache)
  poseclip/                    In-memory .pose compositor (hearing)
extension/                     Manifest V3
  content/camera-hook.js       getUserMedia, canvas, mic
  content/meet.js              HUD, JPEG, STT, chat
  content/skeleton.js          Avatar drawing
  offscreen.js + sandbox.js    Holistic WASM
  lib/                         api, capture, landmarks, prefs
```

No PDF/QR/YouTube lab pipeline lives here. Building `.pose` files is
`avatar-cloud`. Training the GGUF is `semantic-cloud`.

---

## Run

Python 3.9–3.12, Windows, Chrome, Internet (Cloud Run). Weights: `git lfs pull`.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
py -3 packaging/fetch_extension_assets.py   # Holistic WASM → extension/vendor/
python run_backend.py                       # or LSABackend.bat
```

Chrome → `chrome://extensions` → developer mode → load `extension/`.
After editing content scripts, reload the extension **and** the Meet tab.

ILSA: splash checks the translator (up to 4 min, cold start) and T-pose
(20 s). Then: mode → (if signer) hand → Start. The extension popup does not
start the engine. `--no-semantic` skips the translator ping.

Overrides: `LSA_SEMANTIC_URL`, `LSA_AVATAR_URL`, or
`%LOCALAPPDATA%\ILSA\avatar_url.txt`. Defaults in
`src/semantic/config.py` and `src/avatar/config.py`.

**Exe:** `packaging\build_exe.bat` → `extension/bin/ILSA.zip`. The GGUF is not
bundled. Upload: `packaging\upload_ilsa_zip.ps1`.

---

## Local API (`127.0.0.1:8765`)

Meet does not `fetch` loopback (Chrome LNA). The service worker proxies
`lsa-api`. CORS includes `https://meet.google.com`.

| Method | Path | Use |
|--------|------|-----|
| GET | `/health` | `ok`, `mode`, `semantic_ready`, `left_handed` |
| GET | `/config` | Capture thresholds |
| GET | `/metrics` | Session summary |
| POST | `/session` | Reset buffer (handedness is set by ILSA) |
| POST | `/sign` | Frames `{pose, left_hand, right_hand}` |
| POST | `/utterance/end` | Close utterance → Cloud Run `/sign` |
| POST | `/stt` | WAV → Vosk |
| POST | `/hearing` | Spanish → glosses + avatar clip |
| POST | `/activity` | Keep the utterance open |

Cloud Run returns `ms` (inference, RNF-01 ≤ 2 s) and `queued_ms` (queue behind
the `infer_slot` lock). `/sign` and `/hearing` are **not** called at the same
time in a call: one ILSA is one mode. Mixing them in a bench inflates the queue.

Client timeouts: translator 240 s (cold start), T-pose 20 s (health) /
60 s (GET `.pose`). Avatar queue in Meet: 20 clips. Max signs per animated
phrase: 24 (`_MAX_SIGNS`; ~40 s at 30 fps, not an LSA rule).

---

## Metrics

Live: `%LOCALAPPDATA%\ILSA\metrics\live-YYYYMMDD.jsonl` (`sign`, `utterance`,
`hearing`). Fields: `pause_ms` (annotated 4 s), `queued_ms`, `infer_ms`,
`remote_ms` (roundtrip), `delivery_ms` (T-pose + compositor, hearing).

Bench against production Cloud Run (valid signer-dataset glosses, mixed
lengths):

```bash
python scripts/bench_latency.py --channel signer --n 20
python scripts/bench_latency.py --channel hearing --n 20
python scripts/bench_latency.py --channel both --n 15
```

`--channel both` runs signer and **then** hearing. `--interval 2` isolates
inference; `0` is a burst (queue). Hearing includes clip delivery unless
`--no-avatar`. The Meet HUD is not in the bench (signer delivery = 0).

| Stage | What | Where |
|-------|------|-------|
| Pause | Hands away → 4 s → close phrase | Constant; not the model |
| Queue | Request on the server → model takes it | `queued_ms` |
| Inference | Model takes the request → text | `ms` / `infer_ms` |
| Delivery | Text on the client: HUD/voice (signer) or T-pose+canvas (hearing) | `delivery_ms` in hearing |

---

## Camera

`camera-hook.js` calls native `getUserMedia` **once**, keeps the stream on
`#lsa-real-cam`, and gives Meet the canvas. If Meet asks again, it reuses the
track. With LSA off it does not wrap (otherwise Meet says the camera is
blocked). On stop it does not `stop()` real tracks.

On Windows, Media Foundation opens in *controlling mode* (one active client).
Chrome treats Zoom/Teams/Camera as exclusive. Several tabs of the same Chrome
can share. ILSA.exe **does not** open the webcam.
