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
una LLM arma una oración en español. Esa oración se pinta en el video que
Meet ya está enviando.

Hay dos procesos:

1. **Extensión Chrome** (captura, recorte, UI de Meet).
2. **`python run_backend.py`** (clasificador + LLM en `127.0.0.1:8765`).

---

## 1. Contrato HTTP (20 min)

Empezá por lo que une ambos procesos. Es corto y no tiene MediaPipe.

1. `extension/lib/api.js` — cliente: `sign`, `endUtterance`, `health`.
2. `src/backend/server.py` — las mismas rutas del otro lado.
3. `run_backend.py` — cómo arranca el servidor.

Pregunta que tenés que poder responder: *¿qué JSON manda un `POST /sign`?*

---

## 2. Una seña, sin cámara (30 min)

Cómo un montón de puntos 3D se vuelve una etiqueta (`HOLA`, `A`, …).

1. `extension/lib/landmarks.js` — `packFrame`, `anyHandPresent`, vector 225.
2. `src/backend/landmarks_payload.py` — el mismo vector en Python.
3. `src/core/landmarks.py` — ancla en hombros, trim, subsampleo a 16 frames.
4. `src/classifier/config.py` — `MAX_FRAMES`, umbrales, lista de clases.
5. `src/classifier/arch.py` — `TinySkeletonClassifier.forward`.
6. `src/backend/session.py` — `ingest_sign`: cooldown, top-3, buffer.

Pregunta: *¿por qué el tensor es `(1, 16, 225)`?*

Los landmarks **no** los calcula el backend. MediaPipe Holistic corre en
`sandbox.js` (navegador). Al back solo llegan puntos 3D de **una seña ya
recortada**.

---

## 3. Recortar la seña en el tiempo (25 min)

1. `extension/lib/capture.js` — máquina de estados (empieza / gracia / flush).
2. `src/app/utterance.py` — `UtteranceBuffer.try_add`.
3. `src/core/repeat_policy.py` — letras vs dígitos vs glosas léxicas.

Pregunta: *¿qué cierra el enunciado, la seña o el reloj de 4 s?*

---

## 4. Español (20 min)

1. `src/core/repeat_policy.py` otra vez — `format_literal_utterance` (deletreo).
2. `src/semantic/translator.py` — `translate_glosses`.
3. `src/semantic/native_llama.py` — `llama-server.exe` en Windows/3.14.
4. `src/core/conversation_memory.py` — contexto de turnos previos.
5. `LSASession.close_utterance` en `session.py`.

Pregunta: *¿cuándo ni siquiera se llama a la LLM?*

---

## 5. Meet: el video (40 min)

Leé esto **después** de saber qué es una seña. Si no, el hook de cámara
parece magia negra.

1. `extension/manifest.json` — dos content scripts, sandbox, offscreen.
2. `extension/popup.js` — quién dispara el arranque.
3. `extension/background.js` — centralita: popup ↔ Meet ↔ offscreen.
4. `extension/content/inject-gum.js` — intercepta `getUserMedia` (mundo MAIN).
5. `extension/content/meet.js` — HUD + JPEG hacia el worker.
6. `extension/offscreen.js` — Holistic + `capture.js` + `LsaApi`.
7. `extension/sandbox.js` — WASM de MediaPipe (CSP relajado).

Pregunta: *¿por qué Meet ve un canvas y no la cámara cruda solo cuando LSA está ON?*

---

## 6. Escritorio (opcional)

Misma idea, otra UI. Solo si te interesa `python run.py`.

1. `run.py` → `src/app/main.py`
2. `src/app/capture.py` — equivalente de `capture.js`
3. `src/app/workers.py` / `ui.py`

---

## Qué no leer primero

- `extension/vendor/mediapipe/*` — binarios de Google, no son nuestro diseño.
- `src/semantic/bin/` — `llama-server` descargado al vuelo.
- Pesos `.pth` / `.gguf` — artefactos, no código.
- `src/app/semantic_eval.py` y `--probe-semantic` — evaluación, no el live.

---

## Diagrama mínimo (para tenerlo a mano)

```
Meet getUserMedia
    → inject-gum.js  (cámara real + canvas con español)
    → meet.js        (JPEG ~192 px)
    → background.js
    → offscreen.js → sandbox.js (Holistic)
    → capture.js     (16 frames)
    → POST /sign
    → session.py     (TinySkeleton → glosa)
    → pausa 4 s
    → POST /utterance/end
    → translator.py  (español)
    → meet.js HUD + inject-gum subtítulo (caduca 8 s)
```
