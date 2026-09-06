# Intérprete de Lengua de Señas Argentina (LSA)

Este repositorio traduce **señas LSA a español** en tiempo real.

La cámara no envía video al clasificador. MediaPipe Holistic extrae un esqueleto
(pose + dos manos). Ese esqueleto se recorta en **una seña a la vez**, se
clasifica como **glosa** (etiqueta léxica: `HOLA`, `MAMA`, `A`, …) y, cuando la
persona deja de señar unos segundos, una LLM convierte la lista de glosas en
una **oración en español**.

Hay **dos formas de usarlo**, que comparten el mismo clasificador y la misma LLM:

| Uso | Entrada | Cómo se corre |
|-----|---------|----------------|
| App de escritorio | Webcam + ventana OpenCV | `python run.py` |
| Extensión Chrome | Página propia o Google Meet | `python run_backend.py` + extensión descomprimida |

No hace falta haber visto el código antes: el mapa de archivos y el flujo
completo están en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## Qué necesitás

- Windows (el flujo de la LLM nativa está pensado para Windows).
- Python 3.9–3.12 **o** 3.14.
  - En **3.14** no instales `llama-cpp-python`: el backend descarga
    `llama-server.exe` (CPU) la primera vez.
  - En **3.10–3.12** podés usar `llama-cpp-python` si preferís.
- Chrome, si vas a usar la extensión.
- (Opcional) GPU NVIDIA para el clasificador PyTorch. La LLM de la extensión
  corre en **CPU por defecto**.

Los pesos del clasificador (`src/classifier/weights/`) y los modelos `.gguf`
(`src/semantic/outputs/`) suelen ir con **Git LFS**. Después de clonar:

```bash
git lfs pull
```

---

## Instalación de Python

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Si no tenés GPU, instalá la rueda CPU de PyTorch en lugar de `cu124`.

---

## App de escritorio

```bash
python run.py
```

| Comando | Qué hace |
|---------|----------|
| `python run.py` | Cámara, glosas, español y (si está activo) voz |
| `python run.py --no-llm` | Solo glosas; arranque más rápido |
| `python run.py --eval` | Recorre señas de evaluación y escribe un CSV |
| `python run.py --eval-semantic` | Evalúa glosas→español (hold-out) |
| `python run.py --probe-semantic` | Probá glosas escritas, sin cámara |

Teclas en la ventana: `q` sale, `m` cambia el modo de captura, `c` limpia el
contexto de conversación, `n` saltea una seña en modo evaluación.

---

## Extensión Chrome + motor local

La extensión **no** clasifica ni traduce sola. Habla con un FastAPI en
`http://127.0.0.1:8765`.

### 1. Descargar MediaPipe (una vez)

Los WASM/tflite de Holistic no se commitean (pesan mucho). Hay que generarlos:

```bash
py -3 packaging/fetch_extension_assets.py
```

Eso llena `extension/vendor/mediapipe/` y, si faltan, los íconos.

### 2. Arrancar el motor

```bash
python run_backend.py
```

Otras variantes:

```bash
python run_backend.py --no-llm
python run_backend.py --gpu          # LLM con Vulkan si está disponible
python run_backend.py --port 8765
```

La primera vez en Python 3.14 baja `llama-server.exe` a `src/semantic/bin/`
(carpeta ignorada por git).

También podés usar `LSABackend.bat` en la raíz, que llama al mismo script.

### 3. Cargar la extensión

1. Chrome → `chrome://extensions`
2. Activar **Modo de desarrollador**
3. **Cargar descomprimida** → carpeta `extension/`
4. Tras cambiar `manifest.json` o content scripts, **recargar la extensión**.
   Tras cambiar el hook de cámara de Meet, **recargar la pestaña de Meet**.

Al instalar se abre una guía (`welcome.html`). El popup de la extensión abre
el traductor a pantalla completa (`translator.html`) o activa Meet.

### Google Meet

1. Motor en marcha (`run_backend.py`).
2. Entrá a `https://meet.google.com/...` y permití la cámara.
3. Activá LSA desde el popup (o el flujo de Meet de la extensión).

Qué ocurre:

- Un script en el mundo MAIN intercepta `getUserMedia` **solo si LSA está
  habilitado**. Meet recibe un canvas con la imagen de la cámara y el español
  dibujado abajo (espejado para que, con el espejo CSS de Meet, se lea bien).
- Otro script (mundo aislado) muestra un HUD arriba a la derecha: ON mientras
  se está grabando una seña, última glosa, español.
- Los frames JPEG van al *service worker* → documento offscreen → iframe
  sandbox con MediaPipe → `POST /sign`.

El español **desaparece solo a los 8 segundos** (video quemado, HUD y
traductor). Si llega una oración nueva, el reloj se reinicia.

---

## Cómo se decide “una seña”

Alineado entre escritorio y extensión (`src/classifier/config.py` y
`extension/lib/capture.js`):

1. **Empieza** cuando hay al menos una mano usable (modo `auto`).
2. **Termina** cuando el buffer llega a 60 frames **o** ambas manos desaparecen
   durante ~0,4 s (se repite el último frame en esa gracia).
3. Se recorta/submuestrea a **16 frames** y se manda al clasificador.
4. Si la confianza es baja, hay cooldown o la seña es demasiado corta, se
   descarta.
5. Tras ~**4 s** sin actividad de señado se cierra el enunciado y corre la LLM.

El clasificador espera un tensor de forma `(1, 16, 225)`: 16 frames ×
(33 puntos de pose × 3 + 21×3 de cada mano).

---

## Mapa del repositorio

```
run.py                 App de escritorio
run_backend.py         API local para la extensión
LSABackend.bat         Atajo Windows al backend
requirements.txt
packaging/             Scripts para WASM de MediaPipe y (opcional) .exe
extension/             Extensión Manifest V3
src/
  app/                 OpenCV, UI, workers, eval
  backend/             FastAPI + sesión de inferencia
  core/                Landmarks, memoria, política de repeticiones
  classifier/          TinySkeleton + pesos + lista de clases
  semantic/            Prompts, GGUF, llama-server
docs/ARQUITECTURA.md   Pipeline, Meet, API, captura
```

**No forma parte del runtime** (y no se versiona):

- `señario1/`, `señario2/`: láminas PNG de diccionario visual, si las tenés
  en el disco.
- `src/semantic/bin/`: `llama-server` descargado al vuelo.
- `extension/vendor/mediapipe/*.wasm` (y similares): regenerar con el script
  de `packaging/`.

---

## API del motor (`127.0.0.1:8765`)

| Método | Ruta | Uso |
|--------|------|-----|
| GET | `/health` | ¿Clasificador y LLM listos? |
| GET | `/config` | Umbrales de captura |
| GET | `/state` | Glosas pendientes y último español |
| POST | `/session` | Nueva sesión (`left_handed`) |
| POST | `/sign` | Lista de frames de una seña |
| POST | `/activity` | “Sigo señando” (retrasa el cierre) |
| POST | `/utterance/end` | Cerrar enunciado y traducir |
| POST | `/conversation/clear` | Vaciar memoria |
| GET | `/download/exe` | `.exe` o `.bat` si existen |

CORS está abierto a orígenes `chrome-extension://`.

---

## Problemas frecuentes

| Síntoma | Qué probar |
|---------|------------|
| Extensión: “motor no listo” | `python run_backend.py` y recargar el popup |
| Meet: cámara bloqueada | Recargar Meet; LSA solo envuelve `getUserMedia` si está ON |
| Meet: sin esqueleto / sin glosas | `py -3 packaging/fetch_extension_assets.py` y recargar la extensión |
| LLM no carga en Python 3.14 | Normal sin `llama-cpp-python`; esperar la descarga de `llama-server.exe` |
| Subtítulos que no se van | Recargar extensión **y** la pestaña de Meet (content script viejo) |
| Clasificador en CPU lento | Instalar PyTorch CUDA; la LLM sigue en CPU salvo `--gpu` |
