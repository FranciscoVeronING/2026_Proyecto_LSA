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
| **LLM semántica** | Modelo GGUF (por defecto Qwen2.5-3B) que arma una oración en español. |
| **Holistic** | Solución MediaPipe: cuerpo + cara + manos en un solo grafo. Usamos pose y manos. |

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
  → LLM  →  español
  → (escritorio) voz pyttsx3
  → (Meet) texto quemado en el video + HUD
```

El video **nunca** entra a PyTorch. Si MediaPipe falla, el clasificador no
tiene nada que hacer.

### 2.1 Escritorio (`python run.py`)

- OpenCV abre la webcam.
- MediaPipe **Python** (`mediapipe` en `requirements.txt`) corre en el mismo
  proceso.
- `src/app/capture.py` recorta señas.
- `src/app/workers.py` llama al clasificador y a la LLM.
- `src/app/ui.py` dibuja overlay y texto.

Punto de entrada: `run.py` → `src/app/main.py`. Los flags `--eval-semantic` y
`--probe-semantic` evitan importar OpenCV para no gastar VRAM.

### 2.2 Extensión + FastAPI (`python run_backend.py`)

Chrome no puede cargar el `.pth` ni el GGUF. La extensión solo:

1. Detecta landmarks (MediaPipe **WASM** en un iframe sandbox).
2. Decide cuándo cortar la seña (`extension/lib/capture.js`).
3. POST JSON al backend.

El backend (`src/backend/`) es el mismo TinySkeleton + la misma LLM que el
escritorio, envueltos en una sesión (`LSASession`).

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
- Un canvas `captureStream(30)` es lo que Meet muestra como “cámara”.
- El español se dibuja en el canvas. Meet espeja el preview local con CSS; el
  texto se dibuja con `translate(w,0); scale(-1,1)` para que se lea bien.
- Desactivar LSA **no** hace `stop()` de los tracks reales (eso cortaba Meet).
  Solo deja de quemar subtítulos / deja de interceptar si no hay pipeline.

El HUD (`extension/content/meet.js`, mundo **aislado**) no puede tocar el
prototipo de `getUserMedia`. Habla con el hook por `window.postMessage` y con
el service worker por `chrome.runtime`.

Flujo de un frame en Meet:

```
#lsa-real-cam
  → JPEG chico (~192 px, calidad ~0.42, cada ~70 ms)
  → content script
  → background.js
  → offscreen.html (Holistic + capture.js + LsaApi)
  → POST /sign o /utterance/end
  → mensaje lsa-caption
  → HUD + postMessage LSA_CAPTION → canvas de Meet
```

MediaPipe en Meet rinde **pocos FPS**. Por eso el fin de seña no espera
“12 frames sin manos” a 30 fps (tardaría segundos), sino **tiempo**
(`missing_hands_limit / 30` ≈ 0,4 s). Durante la gracia se duplica el último
frame para no acortar el tensor.

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

Modo `auto`: la primera mano usable **empieza** la grabación (no hace falta
esperar 4 frames estáticos). Eso importa porque Meet entrega pocos landmarks
por segundo.

## 6. Clasificador

- Arquitectura: `src/classifier/arch.py` (`TinySkeleton`).
- Pesos: `src/classifier/weights/tinyskeleton_best.pth`.
- Nombres de clase: `mapeo_clases.json` / lista `SIGN_CLASSES` en `config.py`.
- Vector por frame: 33×3 pose + 21×3 mano izq. + 21×3 mano der. = **225**.
- Device: CUDA si PyTorch la ve; si no, CPU.

El backend aplica un suavizado EMA (`LandmarkSmoother`, α≈0.6) y luego arma
el tensor (`sequence_buffer_to_model_input`).

Política de repeticiones (`src/core/repeat_policy.py`): letras pueden repetirse
hasta 2 veces; dígitos sin tope; glosas léxicas no se aceptan dos veces
seguidas.

## 7. Traductor semántico

- Prompt de sistema: `src/semantic/prompts/sys_prompt.txt`.
- Few-shots: `few_shots_examples.json`.
- Modelos: carpetas `src/semantic/outputs/*_gguf/*.gguf` (catálogo en
  `src/semantic/models.py`). Default: `qwen2.5-3b`.
- Memoria de conversación: `src/core/conversation_memory.py` (últimos N
  turnos). Se puede apagar con `USE_CONVERSATION_HISTORY` para eval.

En Python 3.14 Windows, `translator.py` usa `native_llama.py`: proceso
`llama-server.exe` HTTP, no el binding `llama-cpp-python`.

Generación: temperatura baja, 64 tokens máx., sin penalización de repetición
fuerte (si no, se rompen DNI y números).

## 8. Subtítulos que caducan

El español no debe quedar pegado en la videollamada.

- Constante `SUBTITLE_HOLD_MS = 8000` en `inject-gum.js` y `meet.js`.
- Un texto nuevo **reinicia** el temporizador.
- El offscreen **no** reenvía el último español en cada tick de debug (eso
  reseteaba el reloj para siempre). Solo lo manda cuando el backend devuelve
  `spanish` en ese snapshot.

## 9. Archivos de la extensión

| Archivo | Rol |
|---------|-----|
| `manifest.json` | MV3, permisos Meet, sandbox, CSP |
| `background.js` | Service worker: offscreen, reenvío de frames y captions |
| `popup.html` / `popup.js` | Estado del motor y Meet ON/OFF |
| `welcome.html` | Guía post-instalación |
| `sandbox.html` / `sandbox.js` | Holistic WASM |
| `offscreen.html` / `offscreen.js` | Mismo motor de captura para Meet (sin pestaña visible) |
| `lib/api.js` | Cliente HTTP `127.0.0.1:8765` |
| `lib/landmarks.js` | Empaquetado, presencia de manos, dibujo |
| `lib/capture.js` | Máquina de estados de la seña |
| `content/inject-gum.js` | Hook `getUserMedia` + canvas con subtítulo |
| `content/meet.js` / `meet.css` | HUD y envío de JPEG |

Versión actual del manifiesto: ver `extension/manifest.json`.

## 10. Archivos del backend

| Archivo | Rol |
|---------|-----|
| `run_backend.py` | Entry point (ventana IRIS o `--headless`) |
| `src/backend/iris_app.py` | Ventana mínima: encender / apagar / reiniciar / registros |
| `src/backend/server.py` | Rutas FastAPI, CORS, assets MediaPipe |
| `src/backend/session.py` | Carga del modelo, ingestión, cierre de enunciado |
| `src/backend/landmarks_payload.py` | JSON de frames → tensores |

Logs útiles al correr el motor: `POST /sign: N frames`, top-3, added/rejected,
cierre de enunciado.

## 11. Qué no es este repo

- No es un diccionario pedagógico. Carpetas tipo `señario1/` con PNG, si
  existen en el disco, **no las usa el código**.
- No hay servidor en la nube: todo es localhost.
- No hay entrenamiento aquí. Los `.pth` y `.gguf` se asumen ya exportados.
- `packaging/build_exe.bat` puede generar `IRIS.exe` (PyInstaller, sin consola).
  En desarrollo: `python run_backend.py` (ventana) o `--headless`.

## 12. Orden mental para debuggear

1. ¿`GET http://127.0.0.1:8765/health` dice `ok` y `classifier_ready`?
2. ¿La extensión está recargada y es la carpeta `extension/` de este clone?
3. ¿Existen los WASM en `extension/vendor/mediapipe/`?
4. ¿Las manos se ven en el traductor (overlay) o el HUD de Meet pasa a ON?
5. ¿La consola del backend imprime `POST /sign` al bajar las manos?
6. ¿Tras 4 s de pausa aparece español y se borra a los 8 s?
