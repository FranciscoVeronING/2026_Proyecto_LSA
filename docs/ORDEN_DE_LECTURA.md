# Orden de lectura del código

Leé en este orden. Cada paso asume el anterior. No hace falta memorizar
nombres: el objetivo es armar el mapa mental cámara → glosa → español.

Los comentarios en el código (JSDoc / docstrings) describen **qué hace**
cada función, **qué entra** y **qué sale**. Este archivo solo dice **por
dónde empezar**.

---

## 0. Una frase del sistema

Alguien seña frente a Meet. MediaPipe saca un esqueleto. Un recortador
decide “esto es una seña”. TinySkeleton la nombra (glosa). Tras una pausa,
ILSA manda las glosas a Cloud Run y recibe una oración en español. Esa
oración se pinta en el video que Meet ya está enviando.

Hay dos procesos locales más Cloud Run:

1. **Extensión Chrome** (captura, recorte, UI de Meet).
2. **ILSA** (`python run_backend.py` o `ILSA.exe`): clasificador en
   `127.0.0.1:8765`, y cliente HTTP del traductor.

ILSA pide modo (sordo / oyente) y Encender. El GGUF no corre en esta rama.

---

## 1. Contrato HTTP (20 min)

Empezá por lo que une ambos procesos. Es corto y no tiene MediaPipe.

1. `extension/lib/api.js` — cliente: `sign`, `endUtterance`, `hearing`, `stt`, `health`.
2. `src/backend/http_schemas.py` — qué se acepta (tope de frames / puntos).
3. `src/backend/server.py` — las mismas rutas del otro lado.
4. `run_backend.py` — cómo arranca el servidor. La ventana es `iris_app.py`.

Pregunta que tenés que poder responder: *¿qué JSON manda un `POST /sign`?*

---

## 2. Una seña, sin cámara (30 min)

Cómo un montón de puntos 3D se vuelve una etiqueta (`HOLA`, `A`, …).

1. `extension/lib/landmarks.js` — `packFrame`, `anyHandPresent`, vector 225.
2. `src/backend/landmarks_payload.py` — el mismo vector en Python.
3. `src/core/landmarks.py` — ancla en hombros, trim, subsampleo a 16 frames.
4. `src/classifier/config.py` — `MAX_FRAMES`, umbrales, lista de clases.
5. `src/classifier/arch.py` — `TinySkeletonClassifier.forward`.
6. `src/backend/session.py` — `ingest_sign`: cooldown, top-3, buffer (CPU).

Pregunta: *¿por qué el tensor es `(1, 16, 225)`?*

Los landmarks **no** los calcula el backend. MediaPipe Holistic corre en
`sandbox.js` (navegador). Al back solo llegan puntos 3D de **una seña ya
recortada**.

---

## 3. Recortar la seña en el tiempo (25 min)

1. `extension/lib/capture.js` — máquina de estados (empieza / gracia / flush).
2. `src/core/utterance.py` — `UtteranceBuffer.try_add`.
3. `src/core/repeat_policy.py` — letras vs dígitos vs glosas léxicas.

Pregunta: *¿qué cierra el enunciado, la seña o el reloj de 4 s?*

---

## 4. Español (20 min)

1. `src/core/repeat_policy.py` otra vez — `format_literal_utterance` (deletreo).
2. `src/semantic/remote.py` — cliente HTTP de Cloud Run (`POST /sordo`, `/oyente`).
3. `src/core/conversation_memory.py` — contexto de turnos previos.
4. `LSASession.close_utterance` en `session.py`.

Pregunta: *¿cuándo ni siquiera se llama a la LLM?*

---

## 5. Ventana ILSA y Meet (40 min)

Leé esto **después** de saber qué es una seña.

1. `src/backend/iris_app.py` — splash (ping remoto), modo, Encender.
2. `extension/manifest.json` — dos content scripts, sandbox, offscreen.
3. `extension/popup.js` — estado y ajustes (no enciende el motor).
4. `extension/background.js` — centralita: popup ↔ Meet ↔ offscreen.
5. `extension/content/inject-gum.js` — intercepta `getUserMedia` (mundo MAIN).
6. `extension/content/meet.js` — HUD + JPEG hacia el worker.
7. `extension/offscreen.js` — Holistic + `capture.js` + `LsaApi`.
8. `extension/sandbox.js` — WASM de MediaPipe (CSP relajado).

Pregunta: *¿por qué Meet ve un canvas y no la cámara cruda solo cuando ILSA está encendido?*

---

## Qué no leer primero

- `extension/vendor/mediapipe/*` — binarios de Google, no son nuestro diseño.
- Pesos `.pth` — artefactos, no código.
- `dist/` y `extension/bin/*.zip` — salida de PyInstaller.

---

## Diagrama mínimo (para tenerlo a mano)

```
Cloud Run
    → POST /sordo y /oyente

ILSA.exe / run_backend.py
    → splash: ping Cloud Run
    → Encender (modo sordo u oyente)
    → uvicorn 127.0.0.1:8765

Meet getUserMedia
    → inject-gum.js  (cámara real + canvas con español)
    → meet.js        (JPEG)
    → background.js
    → offscreen.js → sandbox.js (Holistic)
    → capture.js
    → POST /sign
    → session.py     (TinySkeleton CPU → glosa)
    → pausa 4 s
    → POST /utterance/end
    → remote.py      (Cloud Run → español)
    → meet.js HUD + inject-gum subtítulo (caduca 8 s)
```
