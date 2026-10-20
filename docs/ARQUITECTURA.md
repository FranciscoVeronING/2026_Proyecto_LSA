# Arquitectura del intérprete LSA

Cómo funciona **este** repositorio cuando ILSA está encendido en Meet.
Complementa el [README](../README.md). Chrome:
[EXTENSION.md](EXTENSION.md). Esqueleto oyente: [AVATAR.md](AVATAR.md).
Cómo se fabricaron los `.pose`: [DICCIONARIO.md](DICCIONARIO.md).

---

## 1. Vocabulario

| Término | Significado aquí |
|---------|------------------|
| **Glosa** | Etiqueta de una seña (`HOLA`, `MAMA`, `A`). No es español fluido. |
| **Enunciado** | Lista de glosas entre dos pausas largas. Eso es lo que ve el traductor. |
| **Landmarks** | Coordenadas de pose (33 puntos) y manos (21 + 21). En el clasificador van normalizadas. |
| **Frame de seña** | Un instante de landmarks, no un JPEG. |
| **TinySkeleton** | Transformer chico: 16 frames × 225 números → una clase. |
| **Traductor / semántico** | FastAPI en Cloud Run: `POST /sign` (glosas→español) y `POST /hearing` (español→glosas). |
| **T-pose** | Clip `.pose` de una glosa en el servidor de movimiento. |
| **Holistic** | MediaPipe: cuerpo + cara + manos. El clasificador usa pose y manos. |
| **ILSA** | Ventana del motor local (`ILSA.exe` o `python run_backend.py`). |
| **MAIN / aislado** | Dos mundos de content scripts en Meet. Ver [EXTENSION.md](EXTENSION.md). |

La cara se detecta pero **no entra** al vector de 225 dimensiones.

---

## 2. Dos modos, un mismo motor

ILSA pide modo **antes** de Encender. No se cambia a mitad de sesión.

| Modo (`_mode`) | Quién | Entrada | Salida en Meet |
|----------------|--------|---------|----------------|
| **Sordo** (`signer`) | Quien seña | Cámara → Holistic → `/sign` | Español (subtítulo; opcional audio TTS local, no hacia la llamada) |
| **Oyente** (`hearing`) | Quien habla | Mic de Meet y/o chat | Glosas pintadas + esqueleto 50/50 |

El popup de la extensión muestra estado y ajustes. **No** enciende el motor.

Con ILSA encendido, Meet arranca solo. Apagar ILSA corta subtítulos y deja de
envolver la cámara. Los tracks reales de Meet **no** se `stop()`: eso cortaba
la llamada.

---

## 3. Pipeline sordo

```
cámara (Meet)
  → inject-gum.js  (video real oculto + canvas que ve Meet)
  → JPEG chico (~320 px)
  → meet.js → background.js → offscreen.js
  → sandbox.js  (Holistic WASM)
  → capture.js  (empieza / gracia / flush)
  → POST /sign   { frames: [ { pose, left_hand, right_hand }, ... ] }
  → landmarks_payload + EMA
  → trim / subsampleo a 16
  → TinySkeleton CPU  →  glosa + confianza
  → UtteranceBuffer + repeat_policy
  → pausa ~4 s  o  POST /utterance/end
  → POST Cloud Run /sign  { glosses } → { spanish }
  → HUD + LSA_CAPTION  (caduca 8 s)
```

El video **nunca** entra a PyTorch. Si Holistic no ve manos, no hay seña.

No se usa movimiento de **píxeles de todo el cuadro** para arrancar: en Meet
hay gente detrás y dispara falsos positivos. Solo manos.

Holistic en Meet rinde pocos FPS. El fin de seña es **manos fuera** ~0,4 s
(`MISSING_HANDS_LIMIT / 30`). Durante la gracia se duplica el último frame.

---

## 4. Pipeline oyente

```
mic de Meet (si no hay mute y captura ≠ solo texto)
  → PCM 16 kHz en MAIN → WAV base64
  → POST /stt  →  Vosk (español, en esta PC)

chat de Meet (Enter / Enviar, si captura ≠ solo voz)
  → texto, sin STT

español
  → POST /hearing
  → POST Cloud Run /hearing  { spanish } → { glosses }
  → GET tpose-server /tposes/{STEM}.pose  (una por glosa / letra)
  → extraction/meet_sequence.py  (trim, park, EMA, puentes)
  → JSON de frames  (p, f, l, r)
  → LSA_CAPTION + LSA_AVATAR
  → canvas 50/50, idle al terminar
```

`POST /hearing` trae `glosses` y `avatar`.

Mute en Meet corta el tap de audio. Cámara apagada no alimenta el pipeline
sordo. Detalle: [EXTENSION.md](EXTENSION.md). Stems y canvas:
[AVATAR.md](AVATAR.md).

No hay memoria de conversación hacia la LLM: cada frase se traduce sola.

---

## 5. Contrato HTTP local

Cliente: `extension/lib/api.js`. Esquemas: `src/backend/http_schemas.py`.
Rutas: `src/backend/server.py`. Arranque: `run_backend.py` → `iris_app.py`
(o `--headless`).

Origen típico del `fetch`: `https://meet.google.com` (content script), no
`chrome-extension://`. Por eso CORS incluye Meet. Header
`Access-Control-Allow-Private-Network` para Chrome reciente.

### `POST /sign`

Lista de frames ya recortados. Cada punto es `{ x, y, z }` (y a veces
confianza). Tope de frames y de puntos: `http_schemas.py`.

Respuesta: snapshot de sesión (glosa aceptada o rechazada, buffer, a veces
español si el cierre coincidió).

En modo oyente responde **400**.

### `POST /stt`

`{ "wav_b64": "...", "lang": "es-ES" }` → `{ "spanish", "ok" }`.
Vosk `vosk-model-small-es-0.42` (se baja a `%LOCALAPPDATA%\ILSA` la primera
vez). El `lang` se ignora: el modelo es español.

### `POST /hearing`

`{ "spanish", "final" }` → snapshot con `glosses`, `caption` y `avatar`.

### `GET /health`

`ok` (sesión creada), `mode`, `classifier_ready`, `semantic_ready`, errores.
El splash de ILSA espera esto más un ping a Cloud Run.

---

## 6. Captura y umbrales (sordo)

Canónico JS: `extension/lib/capture.js`. Python: `src/classifier/config.py`.
La extensión pide un subconjunto con `GET /config`.

| Parámetro | Valor | Rol |
|-----------|-------|-----|
| `CAPTURE_BUFFER_SIZE` | 60 | Máximo de frames crudos de una seña |
| `MAX_FRAMES` | 16 | Lo que ve el modelo (cambiar esto obliga a reentrenar) |
| `MISSING_HANDS_LIMIT` | 12 | Gracia ≈ 12/30 s |
| `MIN_CAPTURE_FRAMES` | 5 | Por debajo, se tira la seña |
| `UTTERANCE_PAUSE_SEC` | 4.0 | Cierra enunciado y llama al traductor |
| `CONFIDENCE_THRESHOLD` | 0.75 | Rechazo del clasificador |
| `INFERENCE_COOLDOWN_SEC` | 1.0 | Evita la misma seña dos veces seguidas |
| `LETTER_MAX_CONSECUTIVE` | 2 | Letras; dígitos sin tope; léxico no se duplica seguido |

Submuestreo: índices `floor(i * (n-1) / (target-1))`. Si hay 16 o menos
frames, se mandan todos.

Modo `auto`: la primera mano usable **empieza** la grabación.

---

## 7. Clasificador

- Arquitectura: `src/classifier/arch.py` (`TinySkeletonClassifier`).
- Pesos: `src/classifier/weights/tinyskeleton_best.pth`.
- Nombres: `mapeo_clases.json` / `SIGN_CLASSES`.
- Vector por frame: 33×3 + 21×3 + 21×3 = **225**.
- En ILSA corre en **CPU** (`CUDA_VISIBLE_DEVICES=""`).
- Suavizado EMA (`LandmarkSmoother`, α≈0.6) y `frames_to_matrix` en
  `src/core/landmarks.py`.
- Alias hacia la LLM: `GLOSS_NORMALIZER` (p. ej. `el_ella` → `EL`). **No**
  unifica género de hermano/hermana para el clasificador; el stem T-pose sí
  recorta `_A` / `_RA` (ver [AVATAR.md](AVATAR.md)).

Política: `src/core/repeat_policy.py`. Letras hasta 2 iguales; dígitos sin
tope; glosas léxicas no dos veces seguidas.

`LSASession.ingest_sign` / `close_utterance`: `src/backend/session.py`.

---

## 8. Traductor semántico

ILSA no carga GGUF. `src/semantic/remote.py`:

`https://ilsa-semantic-service-705664945052.us-central1.run.app`

- Sordo: `POST /sign` `{ "glosses": "..." }` → `{ "spanish": "..." }`.
- Oyente: `POST /hearing` `{ "spanish": "..." }` → `{ "glosses": "..." }`.

Keepalive cada 4 min para no enfriar Cloud Run. `X-LSA-Client` es un UUID en
`%LOCALAPPDATA%\ILSA\client_id.txt`. Token opcional: `LSA_SEMANTIC_TOKEN`.

El código del servidor GGUF vive en la rama `semantic-cloud`.

---

## 9. T-pose (resumen)

`src/avatar/remote.py` hace `GET /tposes/{STEM}.pose` a

`https://tpose-server-705664945052.southamerica-east1.run.app`

Caché en `%LOCALAPPDATA%\ILSA\tposes`. El compositor
(`extraction/meet_sequence.py` + `render_pose_pair.py`) arma frames en esta
PC. Hace falta el paquete `pose-format` (ILSA lo instala solo si falta).
OpenCV solo se usa para mp4 de laboratorio, no en Meet.

Reglas de stem, deletreo, JSON del canvas: [AVATAR.md](AVATAR.md).

---

## 10. Ventana ILSA y empaquetado

| Archivo | Rol |
|---------|-----|
| `run_backend.py` | Entry; mete `src/` en `sys.path` |
| `src/backend/iris_app.py` | Splash, logo PNG, modo, Encender / Apagar, modal CAS |
| `src/backend/uvicorn_handle.py` | Hilo uvicorn (Tk no se cuelga) |
| `src/backend/ui_theme.py` | Colores de la ventana |
| `packaging/build_exe.bat` | `dist/LSABackend/ILSA.exe` + `extension/bin/ILSA.zip` |
| `packaging/LSABackend.spec` | Hiddenimports: classifier, semantic, avatar, vosk |
| `packaging/rthook_cpu.py` | PATH de DLLs de Torch en el exe |

`--headless --mode hearing` sirve para scripts. El API en `0.0.0.0` expone
clasificador y cliente semántico a la LAN: no hacerlo.

Logs útiles: `POST /sign: N frames`, top-3, added/rejected, `/stt → '…'`,
`/hearing … ms`, `GET /tposes/HOLA.pose … ms`.

---

## 11. Orden de lectura del código

Cada paso asume el anterior. Los comentarios en el código dicen **qué hace**
cada función. Acá, **por dónde empezar**.

1. **Contrato HTTP** (~20 min). `extension/lib/api.js` →
   `http_schemas.py` → `server.py` → `run_backend.py`.
   Pregunta: ¿qué JSON manda un `POST /sign`?

2. **Una seña, sin cámara** (~30 min). `landmarks.js` →
   `landmarks_payload.py` → `core/landmarks.py` → `classifier/config.py` →
   `arch.py` → `session.py`.
   Pregunta: ¿por qué el tensor es `(1, 16, 225)`?

3. **Recortar en el tiempo** (~25 min). `capture.js` → `utterance.py` →
   `repeat_policy.py`.
   Pregunta: ¿qué cierra el enunciado, la seña o el reloj de 4 s?

4. **Español** (~20 min). `repeat_policy.format_literal_utterance` →
   `semantic/remote.py` → `LSASession.close_utterance`.
   Pregunta: ¿cuándo ni siquiera se llama al remoto?

5. **Meet** (~40 min). `iris_app.py` → `manifest.json` → `popup.js` →
   `background.js` → `inject-gum.js` → `meet.js` → `offscreen.js` →
   `sandbox.js`. Luego [EXTENSION.md](EXTENSION.md) y [AVATAR.md](AVATAR.md).
   Pregunta: ¿por qué Meet ve un canvas solo con ILSA encendido?

**No leer primero:** `extension/vendor/mediapipe/*`, pesos `.pth`, `dist/`,
caché Unsloth en `src/semantic/`.

---

## 12. Qué no es este repo

- No es un diccionario pedagógico en PNG.
- El clasificador corre en ILSA. La LLM corre en Cloud Run. El video no sale
  de la PC de Meet.
- No hay entrenamiento de TinySkeleton aquí.
- `docs/archivo/` es historia de entregas. Si choca con este archivo, gana
  este archivo.

---

## 13. Orden mental para debuggear

1. ¿Responde el `/health` de Cloud Run semántico desde esta PC?
2. ¿Elegiste modo y Encendiste? `GET http://127.0.0.1:8765/health` → `ok`.
3. ¿La extensión es la carpeta `extension/` de este clone, recargada?
4. ¿Existen los WASM en `extension/vendor/mediapipe/`?
5. ¿Meet está recargado después de tocar `inject-gum.js`?
6. ¿La consola del motor imprime `/sign` al bajar las manos (sordo)?
7. ¿Tras ~4 s de pausa aparece español y se borra a los 8 s?
8. ¿Oyente: `/stt` y `/hearing` en el log, y `GET /tposes/…` 200?
