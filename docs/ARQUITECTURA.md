# Arquitectura del intérprete LSA

Documento para alguien que abre el repo por primera vez. Complementa el
[README](../README.md) con el *por qué* de cada pieza.

**Orden de lectura del código** (qué archivo abrir primero):
[`ORDEN_DE_LECTURA.md`](ORDEN_DE_LECTURA.md).

## 1. Vocabulario

| Término | Significado aquí |
|---------|------------------|
| **Glosa** | Etiqueta de una seña reconocida (`HOLA`, `MAMA`, `A`). No es español fluido. |
| **Enunciado** | Lista de glosas entre dos pausas largas. Eso es lo que ve la LLM. |
| **Landmarks** | Coordenadas normalizadas de pose (33 puntos) y manos (21 + 21). |
| **Frame de seña** | Un instante de landmarks, no un JPEG. |
| **TinySkeleton** | Red chica (transformer) que clasifica una secuencia de 16 frames. |
| **Traductor** | FastAPI en Cloud Run (`POST /sordo` y `POST /oyente`). |
| **Holistic** | Solución MediaPipe: cuerpo + cara + manos en un solo grafo. Usamos pose y manos. |
| **ILSA** | Ventana del motor local (`ILSA.exe` o `python run_backend.py`). |

La cara se detecta pero **no entra** al vector de 225 dimensiones del
clasificador.

## 2. Pipeline de punta a punta

```
cámara
  → MediaPipe Holistic (esqueleto)
  → detector de “empieza / termina la seña”
  → 16 frames × 225 números
  → TinySkeleton  →  glosa + confianza
  → buffer de glosas (anti-repetición, cooldown)
  → pausa ~4 s
  → POST /sordo (Cloud Run)  →  español
  → (Meet) texto quemado en el video + HUD
```

El video **nunca** entra a PyTorch. Si MediaPipe falla, el clasificador no
tiene nada que hacer.

### 2.1 Extensión + FastAPI (`python run_backend.py` o `ILSA.exe`)

Chrome no puede cargar el `.pth` ni el GGUF. La extensión solo:

1. Detecta landmarks (MediaPipe **WASM** en un iframe sandbox).
2. Decide cuándo cortar la seña (`extension/lib/capture.js`).
3. POST JSON al backend.

El backend (`src/backend/`) envuelve TinySkeleton y el cliente de Cloud Run
en una sesión (`LSASession` o `HearingSession`).

ILSA tiene dos modos, elegidos en la ventana **antes** de Encender:

| Modo | Quién | Qué hace |
|------|--------|----------|
| **Sordo** | Quien seña | LSA → español (clasificador + Llama 1B). |
| **Oyente** | Quien habla | Voz (mic de Meet) o chat → glosas pintadas en la cámara. |

Con ILSA encendido, Meet arranca la traducción solo. El popup muestra estado
y ajustes; no enciende el motor.

## 3. Por qué un sandbox en la extensión

Manifest V3 bloquea `blob:` / `unsafe-eval` en páginas de extensión.
MediaPipe Holistic **necesita** crear workers desde blobs.

Solución:

- `sandbox.html` + `sandbox.js` corren Holistic con CSP relajado.
- `offscreen.html` manda un JPEG al iframe y recibe
  `{ pose, left_hand, right_hand }`.
- Los `.wasm` / `.tflite` viven en `extension/vendor/mediapipe/` (gitignored;
  se bajan con `packaging/fetch_extension_assets.py`).

No uses `holistic.close()` en cada stop de cámara: recargar el WASM es lento
y frágil. Se reutiliza el grafo.

## 4. Google Meet

Meet pide la cámara con `navigator.mediaDevices.getUserMedia`. Si envolvemos
ese método **siempre**, Meet puede ver la cámara “bloqueada” o un pipeline
muerto.

Reglas del hook (`extension/content/inject-gum.js`, mundo **MAIN**,
`document_start`):

- Solo envuelve cuando LSA está **habilitado**.
- Crea un `<video id="lsa-real-cam">` oculto con el stream real.
- Un canvas `captureStream` es lo que Meet muestra como “cámara”.
- El español se dibuja **abajo** del canvas. Meet espeja el preview local con
  CSS; el texto se dibuja espejado para que se lea bien.
- En modo oyente, el reconocimiento de voz corre en ese mismo mundo MAIN (no
  hay TTS de la transcripción).
- Desactivar LSA **no** hace `stop()` de los tracks reales (eso cortaba Meet).
  Solo deja de quemar subtítulos / deja de interceptar si no hay pipeline.

El HUD (`extension/content/meet.js`, mundo **aislado**) no puede tocar el
prototipo de `getUserMedia`. Habla con el hook por `window.postMessage` y con
el service worker por `chrome.runtime`. Si se recarga la extensión con Meet
abierto, el content script corta el poll (`Extension context invalidated`).

Flujo de un frame en Meet (modo sordo):

```
#lsa-real-cam
  → JPEG (~320 px, calidad ~0.62)
  → content script
  → background.js
  → offscreen.html (Holistic + capture.js + LsaApi)
  → POST /sign o /utterance/end
  → mensaje lsa-caption
  → HUD + postMessage LSA_CAPTION → canvas de Meet
```

MediaPipe en Meet rinde **pocos FPS**. El fin de seña es **manos fuera de
cámara** ~0,4 s (`missing_hands_limit / 30`). No se corta por manos quietas
(señas estáticas). Durante la gracia se duplica el último frame.

No se usa movimiento de **píxeles de todo el cuadro** para arrancar una seña:
en Meet hay gente moviéndose atrás y dispara falsos positivos. Solo manos.

## 5. Captura: empezar, cortar, muestrear

Código canónico JS: `extension/lib/capture.js`.
Umbrales Python: `src/classifier/config.py`.

| Parámetro | Valor | Rol |
|-----------|-------|-----|
| `CAPTURE_BUFFER_SIZE` | 60 | Máximo de frames crudos de una seña |
| `MAX_FRAMES` | 16 | Lo que ve el modelo (submuestreo uniforme) |
| `MISSING_HANDS_LIMIT` | 12 | En JS: gracia en segundos ≈ 12/30 |
| `MIN_CAPTURE_FRAMES` | 5 | Por debajo, se tira la seña |
| `UTTERANCE_PAUSE_SEC` | 4.0 | Cierra enunciado y llama a la LLM |
| `CONFIDENCE_THRESHOLD` | 0.75 | Rechazo del clasificador |
| `INFERENCE_COOLDOWN_SEC` | 1.0 | Evita la misma seña dos veces seguidas |

Submuestreo: índices `floor(i * (n-1) / (target-1))`, igual que un `linspace`
entero. Si hay 16 o menos frames, se mandan todos.

Modo `auto`: la primera mano usable **empieza** la grabación.

## 6. Clasificador

- Arquitectura: `src/classifier/arch.py` (`TinySkeleton`).
- Pesos: `src/classifier/weights/tinyskeleton_best.pth`.
- Nombres de clase: `mapeo_clases.json` / lista `SIGN_CLASSES` en `config.py`.
- Vector por frame: 33×3 pose + 21×3 mano izq. + 21×3 mano der. = **225**.
- En el motor ILSA (`LSASession`) el clasificador corre en **CPU**.

El backend aplica un suavizado EMA (`LandmarkSmoother`, α≈0.6) y luego arma
el tensor (`frames_to_matrix`).

Política de repeticiones (`src/core/repeat_policy.py`): letras pueden repetirse
hasta 2 veces; dígitos sin tope; glosas léxicas no se aceptan dos veces
seguidas.

## 7. Traductor semántico

ILSA no carga GGUF. `src/semantic/remote.py` hace HTTP a Cloud Run:

`https://ilsa-semantic-service-705664945052.us-central1.run.app`

- Sordo: `POST /sordo` con `{ "glosses": "..." }` → `{ "spanish": "..." }`.
- Oyente: `POST /oyente` con `{ "spanish": "..." }` → `{ "glosses": "..." }`.
- Esqueleto oyente: `src/avatar/remote.py` pide
  `GET /tposes/{GLOSA}.pose` a
  `https://tpose-server-705664945052.southamerica-east1.run.app`.
  El stem es igualdad de nombre (`avatar.names.tpose_stem`).
  Los `.pose` no viven en esta PC.

El código del servidor GGUF vive en la rama `semantic-cloud`, no acá.

Para publicar el exe: `packaging\build_exe.bat` y `packaging\upload_ilsa_zip.ps1`.

## 8. Subtítulos que caducan

El español no debe quedar pegado en la videollamada.

- Constante `SUBTITLE_HOLD_MS = 8000` en `inject-gum.js` y `meet.js`.
- Un texto nuevo **reinicia** el temporizador.
- El offscreen **no** reenvía el último español en cada tick de debug. Solo
  lo manda cuando el backend devuelve `spanish` en un cierre de enunciado.

## 9. Archivos de la extensión

| Archivo | Rol |
|---------|-----|
| `manifest.json` | MV3, permisos Meet, sandbox, CSP |
| `background.js` | Service worker: offscreen, reenvío de frames y captions |
| `popup.html` / `popup.js` | Estado del motor, modo, salida Meet, landmarks (debug) |
| `welcome.html` | Guía: descargar ILSA y comprobar `/health` |
| `sandbox.html` / `sandbox.js` | Holistic WASM |
| `offscreen.html` / `offscreen.js` | Captura para Meet (sin pestaña visible) |
| `lib/api.js` | Cliente HTTP `127.0.0.1:8765` |
| `lib/prefs.js` | Preferencias (salida, landmarks) |
| `lib/landmarks.js` | Empaquetado, presencia de manos, dibujo |
| `lib/capture.js` | Máquina de estados de la seña |
| `content/inject-gum.js` | Hook `getUserMedia` + canvas con subtítulo + voz (oyente) |
| `content/meet.js` / `meet.css` | HUD y envío de JPEG |

Versión actual del manifiesto: ver `extension/manifest.json`.

## 10. Archivos del backend

| Archivo | Rol |
|---------|-----|
| `run_backend.py` | Entry point (ventana ILSA o `--headless`) |
| `src/backend/iris_app.py` | Splash (ping remoto), modo sordo/oyente, Encender / Apagar |
| `src/backend/uvicorn_handle.py` | Hilo uvicorn (sin colgar Tk) |
| `src/backend/http_schemas.py` | Límites de JSON (`POST /sign`, etc.) |
| `src/backend/server.py` | Rutas FastAPI; CORS: Meet + `chrome-extension://` + localhost |
| `src/backend/session.py` | Clasificador CPU, ingestión, cierre de enunciado, HTTP /sordo |
| `src/backend/hearing_session.py` | Sesión oyente (voz → POST /oyente) |
| `src/backend/landmarks_payload.py` | JSON de frames → tensores |
| `src/semantic/remote.py` | Cliente HTTP de Cloud Run |

Logs útiles al correr el motor: `POST /sign: N frames`, top-3, added/rejected,
cierre de enunciado.

El API escucha en loopback (`127.0.0.1`). No bindear `0.0.0.0`.

## 11. Empaquetado

- `packaging/build_exe.bat` genera `dist/LSABackend/ILSA.exe` y
  `extension/bin/ILSA.zip` (PyInstaller, sin consola).
  El zip no se versiona.
- `packaging/LSABackend.spec` empaqueta clasificador y runtime CPU.
- En desarrollo: `python run_backend.py` (ventana) o `--headless`.
- Semántico: Cloud Run; este repo solo tiene el cliente HTTP.

## 12. Qué no es este repo

- No es un diccionario pedagógico. Carpetas tipo `señario1/` con PNG, si
  existen en el disco, **no las usa el código**.
- El clasificador corre en ILSA. La LLM corre en Cloud Run. El video no sale
  de la PC de Meet.
- No hay entrenamiento aquí. Los `.pth` se asumen ya exportados.

## 13. Orden mental para debuggear

1. ¿Responde el `/health` de Cloud Run desde esta PC?
2. ¿Elegiste modo y Encendiste? `GET http://127.0.0.1:8765/health` → `ok`.
3. ¿La extensión está recargada y es la carpeta `extension/` de este clone?
4. ¿Existen los WASM en `extension/vendor/mediapipe/`?
5. ¿Meet está recargado después de tocar `inject-gum.js`?
6. ¿La consola del motor imprime `/sign` al bajar las manos?
7. ¿Tras ~4 s de pausa aparece español y se borra a los 8 s?
