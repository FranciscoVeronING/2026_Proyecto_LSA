# ILSA — rama `extension`

Intérprete de LSA en Google Meet: extensión Chrome + motor local. Traduce
**LSA ↔ español** en la llamada. El video no sale de la PC. A la nube van
glosas o español, nunca JPEG.

Esta rama es el **cliente**. El GGUF vive en Cloud Run (`semantic-cloud`).
Los `.pose` viven en Cloud Run (`avatar-cloud` / `tpose-server`).

```
Chrome (extensión)
  Meet: cámara, mic, chat, Holistic WASM, canvas, HUD
        │ HTTP 127.0.0.1:8765  (el service worker proxya lsa-api;
        │                       Meet no hace fetch a loopback)
        ▼
ILSA (ventana / ILSA.exe)
  Sordo:  TinySkeleton CPU → pausa 4 s → POST /sign
  Oyente: Vosk y/o chat → POST /hearing → T-pose + compositor
        │                              │
        ▼                              ▼
  ilsa-semantic-service          tpose-server
  us-central1                    southamerica-east1
  POST /sign  POST /hearing      GET /tposes/{STEM}.pose
```

Meet, con ILSA encendido, **no** envía el stream crudo: envía
`canvas.captureStream`. La cámara física se abre **una sola vez** (Windows
suele tomar el device en modo exclusive; un segundo `getUserMedia` falla con
`NotReadableError`).

---

## Modos

Elegí el modo **antes** de Encender. No se cambia a mitad de sesión (Reiniciar).

| Modo | Quién | Entrada | Salida en Meet |
|------|--------|---------|----------------|
| **Sordo** | Quien seña | Cámara → Holistic → `POST /sign` | Español abajo del video (TTS local opcional, no va a la llamada) |
| **Oyente** | Quien habla / escribe | Mic de Meet y/o chat | Glosas pintadas + esqueleto a la derecha |

En sordo, **Zurdo / Diestro** (Diestro por defecto). El clasificador se entrenó
diestro: zurdo espeja X y cruza las manos.

**Seña (sordo).** Empieza con una mano usable. Termina a 60 frames o cuando
faltan las dos manos ~0,4 s. Se recorta a 16×225 (pose+manos; sin cara).
Confianza baja o repetición no entra al enunciado. A los **4 s** sin señar se
cierra y va al traductor.

**Oyente.** El tap de audio **no** pide otro `getUserMedia`: copia el track de
Meet. Una palabra suelta no pasa por Cloud Run (va derecho al T-pose). Una
oración sí (`POST /hearing`).

---

## Mapa

```
run_backend.py                 Arranque (ventana o --headless)
LSABackend.bat                 Atajo Windows
requirements.txt               pip del motor (sin MediaPipe)
assets/                        Logo
packaging/                     PyInstaller, WASM Holistic, íconos, zip
scripts/bench_latencia.py      Banco cola / inferencia / entrega
src/
  backend/                     API FastAPI 8765 + ventana Tk
    window.py                  Splash (ping semántico + T-pose), modo, mano
    server.py                  Rutas
    signer_session.py          Clasificador + buffer + /sign remoto
    hearing_session.py         Vosk ya vino; /hearing + clips
    metrics.py                 JSONL en %LOCALAPPDATA%\ILSA\metrics
  classifier/                  TinySkeleton + weights/
  core/                        Landmarks, enunciado, repetición
  semantic/                    Cliente HTTP Cloud Run
  avatar/                      Cliente T-pose (stems, GET, caché)
  poseclip/                    Compositor de .pose en RAM (oyente)
extension/                     Manifest V3
  content/camera-hook.js       getUserMedia, canvas, mic
  content/meet.js              HUD, JPEG, STT, chat
  content/skeleton.js          Dibujo del avatar
  offscreen.js + sandbox.js    Holistic WASM
  lib/                         api, capture, landmarks, prefs
```

No hay laboratorio de PDF/QR/YouTube en esta rama. Fabricar `.pose` es
`avatar-cloud`. Entrenar el GGUF es `semantic-cloud`.

---

## Correr

Python 3.9–3.12, Windows, Chrome, Internet (Cloud Run). Pesos:
`git lfs pull`.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
py -3 packaging/fetch_extension_assets.py   # WASM Holistic → extension/vendor/
python run_backend.py                       # o LSABackend.bat
```

Chrome → `chrome://extensions` → modo desarrollador → cargar `extension/`.
Tras tocar content scripts, recargar la extensión **y** la pestaña de Meet.

ILSA: splash comprueba traductor (hasta 4 min, arranque en frío) y T-pose
(20 s). Después: modo → (si sordo) mano → Encender. El popup de la extensión
no enciende el motor. `--no-semantic` saltea el ping al traductor.

Overrides: `LSA_SEMANTIC_URL`, `LSA_AVATAR_URL`, o
`%LOCALAPPDATA%\ILSA\avatar_url.txt`. Defaults en
`src/semantic/config.py` y `src/avatar/config.py`.

**Exe:** `packaging\build_exe.bat` → `extension/bin/ILSA.zip`. El GGUF no viaja.
Subida al release: `packaging\upload_ilsa_zip.ps1`.

---

## API local (`127.0.0.1:8765`)

Meet no hace `fetch` a loopback (Chrome LNA). El service worker proxya
`lsa-api`. CORS incluye `https://meet.google.com`.

| Método | Ruta | Uso |
|--------|------|-----|
| GET | `/health` | `ok`, `mode`, `semantic_ready`, `left_handed` |
| GET | `/config` | Umbrales de captura |
| GET | `/metrics` | Resumen de la sesión |
| POST | `/session` | Reset de buffer (la mano la fija ILSA) |
| POST | `/sign` | Frames `{pose, left_hand, right_hand}` |
| POST | `/utterance/end` | Cierra enunciado → Cloud Run `/sign` |
| POST | `/stt` | WAV → Vosk |
| POST | `/hearing` | Español → glosas + clip avatar |
| POST | `/activity` | Mantiene el enunciado abierto |

Cloud Run semántico responde `ms` (inferencia, RNF-01 ≤ 2 s) y `queued_ms`
(cola del candado `infer_slot`). `/sign` y `/hearing` **no** se piden a la vez
en una llamada: un ILSA es un modo. Mezclarlos en un banco infla la cola.

Timeouts del cliente: traductor 240 s (cold start), T-pose 20 s (health) /
60 s (GET `.pose`). Cola de avatar en Meet: 20 clips. Tope de señas por frase
animada: 24 (`_MAX_SIGNS`; alinea con ~40 s a 30 fps, no es un límite LSA).

---

## Métricas

En vivo: `%LOCALAPPDATA%\ILSA\metrics\live-YYYYMMDD.jsonl` (`sign`,
`utterance`, `hearing`). Campos: `pause_ms` (4 s anotados), `queued_ms`,
`infer_ms`, `remote_ms` (roundtrip), `delivery_ms` (T-pose + compositor,
oyente).

Banco contra Cloud Run de producción (glosas **válidas** del dataset sordo,
distintas longitudes):

```bash
python scripts/bench_latencia.py --canal sordo --n 20
python scripts/bench_latencia.py --canal oyente --n 20
python scripts/bench_latencia.py --canal ambos --n 15
```

`--canal ambos` corre sordo y **después** oyente. `--intervalo 2` aísla
inferencia; `0` es ráfaga (cola). Oyente incluye entrega de clips salvo
`--sin-avatar`. El HUD de Meet no entra en el banco (entrega sordo = 0).

| Etapa | Qué es | Dónde se mide |
|-------|--------|----------------|
| Pausa | Manos fuera → 4 s → cierra frase | Constante; no es el modelo |
| Cola | Pedido en el servidor → el modelo lo toma | `queued_ms` |
| Inferencia | El modelo toma el pedido → texto | `ms` / `infer_ms` |
| Entrega | Texto en el cliente: HUD/voz (sordo) o T-pose+canvas (oyente) | `delivery_ms` en oyente |

---

## Cámara

`camera-hook.js` llama al `getUserMedia` nativo **una vez**, deja el stream en
`#lsa-real-cam` y le da a Meet el canvas. Si Meet pide de nuevo, reusa el
track. Con LSA apagado no envuelve (si no, Meet dice cámara bloqueada). Al
apagar no hace `stop()` de los tracks reales.

En Windows, Media Foundation abre en *controlling mode* (un cliente activo).
Chrome trata a Zoom/Teams/Cámara como exclusive. Varias pestañas del mismo
Chrome sí pueden compartir. ILSA.exe **no** abre la webcam.
