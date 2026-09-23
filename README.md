# Intérprete de Lengua de Señas Argentina (LSA)

Interpretación de LSA en tiempo real. **Empezá a leer el código por**
[`docs/ORDEN_DE_LECTURA.md`](docs/ORDEN_DE_LECTURA.md).

Este repositorio traduce **señas LSA a español** (y, en modo oyente, voz a
subtítulos en Meet).

La cámara no envía video al clasificador. MediaPipe Holistic extrae un esqueleto
(pose + dos manos). Ese esqueleto se recorta en **una seña a la vez**, se
clasifica como **glosa** (etiqueta léxica: `HOLA`, `MAMA`, `A`, …) y, cuando la
persona deja de señar unos segundos, un servidor semántico (Llama 3.2 1B en
**tu PC**, publicado con ngrok) convierte la lista de glosas
en una **oración en español**. ILSA en la laptop de Meet **solo clasifica**.

Hay **dos formas de usarlo**, que comparten el mismo clasificador y la misma LLM:

| Uso | Entrada | Cómo se corre |
|-----|---------|----------------|
| App de escritorio | Webcam + ventana OpenCV | `python run.py` |
| Extensión Chrome + ILSA | Google Meet | `python run_backend.py` o `ILSA.exe`, más la extensión descomprimida |

No hace falta haber visto el código antes: el mapa de archivos y el flujo
completo están en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## Qué necesitás

- Windows (el flujo nativo de la LLM está pensado para Windows).
- Python 3.9–3.12 **o** 3.14.
  - En **3.14** no instales `llama-cpp-python`: el servidor semántico descarga
    `llama-server.exe` (CPU) si hace falta.
  - En **3.10–3.12** podés usar `llama-cpp-python` (rueda CPU).
- Chrome, si vas a usar la extensión.
- El GGUF de sordo en `src/semantic/models/sordo/` (oyente opcional en
  `src/semantic/models/oyente/`).
- `LSA_SEMANTIC_URL` no hace falta: ILSA la baja del release `semantic_url.txt`. Ver
  [`docs/SEMANTICO_NUBE.md`](docs/SEMANTICO_NUBE.md).

Los pesos del clasificador (`src/classifier/weights/`) suelen ir con **Git LFS**.
Después de clonar:

```bash
git lfs pull
```

---

## Instalación de Python

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

En 3.10–3.12, si querés el binding de llama.cpp:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

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

## Extensión Chrome + motor ILSA

La extensión **no** clasifica ni traduce sola. Habla con FastAPI en
`http://127.0.0.1:8765`.

### 1. Descargar MediaPipe (una vez)

Los WASM/tflite de Holistic no se commitean. Hay que generarlos:

```bash
py -3 packaging/fetch_extension_assets.py
```

Eso llena `extension/vendor/mediapipe/` y, si faltan, los íconos.

### 2. Motor ILSA

**Quien usa el exe:** en la extensión, **Instalar motor → Descargar ILSA**.
Descomprimí, abrí **ILSA.exe**. La URL del traductor sale del release (`semantic_url.txt`).
Elegí modo (Sordo u Oyente) y **Encender**. Dejá esa ventana abierta.

**Quien publica el zip** (en esta PC, una vez):

```bash
packaging\build_exe.bat
powershell -File packaging\upload_ilsa_zip.ps1
```

El zip no incluye el GGUF. El traductor es `python run_semantic_server.py` + túnel.

**Quien desarrolla:**

```bash
python run_backend.py
```

Misma ventana, sin empaquetar. Para generar el zip del exe (sin el GGUF):

```bash
packaging\build_exe.bat
```

Para publicar el GGUF en GitHub Releases:

```bash
powershell -File packaging\upload_llama_gguf.ps1
```

Hace falta GitHub CLI autenticado (`gh auth login`).

### 3. Cargar la extensión

1. Chrome → `chrome://extensions`
2. Activar **Modo de desarrollador**
3. **Cargar descomprimida** → carpeta `extension/`
4. Tras cambiar `manifest.json` o content scripts, **recargar la extensión**.
   Tras cambiar el hook de cámara de Meet, **recargar la pestaña de Meet**.

Al instalar se abre una guía (`welcome.html`). El popup muestra si el motor
está conectado y, en modo sordo, la salida en Meet (subtítulo / audio / ambos).

### Google Meet

1. ILSA abierto y Encendido.
2. Entrá a `https://meet.google.com/...` y permití la cámara.
3. Con el motor sano, la traducción arranca sola. Para pararla, **Apagá** en
   el exe.

Qué ocurre:

- Un script en el mundo MAIN intercepta `getUserMedia` **solo si LSA está
  habilitado**. Meet recibe un canvas con la imagen de la cámara y el español
  dibujado abajo.
- Otro script (mundo aislado) muestra un HUD: ON mientras se graba una seña,
  última glosa, español.
- En modo sordo los JPEG van al *service worker* → offscreen → sandbox
  MediaPipe → `POST /sign`.
- En modo oyente el reconocimiento de voz corre en MAIN y el texto se pinta
  en el mismo canvas.

El español **desaparece solo a los 8 segundos**. Si llega una oración nueva,
el reloj se reinicia.

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
run_backend.py         Ventana ILSA + API local
LSABackend.bat         Atajo Windows al backend
requirements.txt
packaging/             WASM MediaPipe, PyInstaller, túnel ngrok/cloudflared
extension/             Extensión Manifest V3
src/
  app/                 OpenCV, UI, workers, eval
  backend/             FastAPI, ventana ILSA, sesión
  core/                Landmarks, memoria, política de repeticiones
  classifier/          TinySkeleton + pesos + lista de clases
  semantic/            Cliente remoto + servidor GGUF (`run_semantic_server.py`)
docs/ARQUITECTURA.md   Pipeline, Meet, API, captura
```

**No forma parte del runtime versionado:**

- `señario1/`, `señario2/`: láminas PNG, si las tenés en el disco.
- `src/semantic/bin/`: `llama-server` descargado al vuelo.
- `extension/vendor/mediapipe/*.wasm` (y similares): regenerar con `packaging/`.
- `extension/bin/ILSA.zip` y `dist/`: salida de PyInstaller.
- `%LOCALAPPDATA%\ILSA\models\`: GGUF descargado en el equipo.

---

## API del motor (`127.0.0.1:8765`)

| Método | Ruta | Uso |
|--------|------|-----|
| GET | `/health` | ¿Motor, clasificador y traductor listos? |
| GET | `/metrics` | Tiempos y conteos de la sesión (JSON) |
| GET | `/config` | Umbrales de captura |
| GET | `/state` | Glosas pendientes y último español |
| GET | `/semantic/models` | Estado del traductor remoto |
| POST | `/session` | Nueva sesión (`left_handed`) |
| POST | `/sign` | Lista de frames de una seña |
| POST | `/hearing` | Modo oyente: español → glosas |
| POST | `/activity` | “Sigo señando” (retrasa el cierre) |
| POST | `/utterance/end` | Cerrar enunciado y traducir |
| POST | `/conversation/clear` | Vaciar memoria |
| GET | `/download/exe` | `.exe` o `.bat` si existen en esta PC |

CORS: `https://meet.google.com`, localhost y orígenes `chrome-extension://`.
El API debe escuchar solo en loopback.

---

## Problemas frecuentes

| Síntoma | Qué probar |
|---------|------------|
| Extensión: “motor apagado” | Abrí ILSA, elegí modo, Encender; recargá el popup |
| ILSA arranca sin traducir | En la PC del GGUF: `run_semantic_server.py` + ngrok; subí `semantic_url.txt` al release |
| Meet: cámara bloqueada | Recargar Meet; LSA solo envuelve `getUserMedia` si ILSA está ON |
| Meet: sin esqueleto / sin glosas | `py -3 packaging/fetch_extension_assets.py` y recargar la extensión |
| LLM no carga en Python 3.14 | Esperar `llama-server.exe` (CPU) |
| Subtítulos que no se van | Recargar extensión **y** la pestaña de Meet |
| `Extension context invalidated` | Recargaste la extensión con Meet abierto: recargá Meet |
