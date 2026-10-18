# Intérprete de Lengua de Señas Argentina (LSA)

Interpretación de LSA en tiempo real. **Empezá a leer el código por**
[`docs/ORDEN_DE_LECTURA.md`](docs/ORDEN_DE_LECTURA.md).

Este repositorio traduce **señas LSA a español** en Google Meet (extensión
Chrome + motor ILSA). En modo oyente, voz o chat → glosas pintadas en el video.

La cámara no envía video al clasificador. MediaPipe Holistic extrae un esqueleto
(pose + dos manos). Ese esqueleto se recorta en **una seña a la vez**, se
clasifica como **glosa** (etiqueta léxica: `HOLA`, `MAMA`, `A`, …) y, cuando la
persona deja de señar unos segundos, ILSA manda las glosas a Cloud Run
y recibe una **oración en español**. ILSA en la laptop de Meet **solo clasifica**.

Se corre con `python run_backend.py` o `ILSA.exe`, más la extensión
descomprimida. El mapa de archivos está en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## Qué necesitás

- Windows (ILSA.exe se arma con PyInstaller en esta PC).
- Python 3.9–3.12 (desarrollo) o el exe empaquetado.
- Chrome, si vas a usar la extensión.
- El traductor semántico vive en Cloud Run
  (`https://ilsa-semantic-service-705664945052.us-central1.run.app`).
  No hace falta GGUF ni `LSA_SEMANTIC_URL` en la laptop de Meet.

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
Descomprimí, abrí **ILSA.exe**. El traductor es Cloud Run (está hardcodeado).
Elegí modo (Sordo u Oyente) y **Encender**. Dejá esa ventana abierta.

**Quien publica el zip** (en esta PC, una vez):

```bash
packaging\build_exe.bat
powershell -File packaging\upload_ilsa_zip.ps1
```

El zip es el clasificador. El GGUF corre en Cloud Run, no en este repo.

**Quien desarrolla:**

```bash
python run_backend.py
```

Misma ventana, sin empaquetar. Para generar el zip del exe:

```bash
packaging\build_exe.bat
```

Hace falta GitHub CLI autenticado (`gh auth login`) para subir el zip.

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
- En modo oyente el mic de Meet (y el chat) van a `POST /stt` / `POST /hearing`;
  las glosas se pintan en el video.

El español **desaparece solo a los 8 segundos**. Si llega una oración nueva,
el reloj se reinicia.

---

## Cómo se decide “una seña”

Alineado entre `src/classifier/config.py` y `extension/lib/capture.js`:

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
run_backend.py         Ventana ILSA + API local
LSABackend.bat         Atajo Windows al backend
requirements.txt
packaging/             WASM MediaPipe, PyInstaller
extension/             Extensión Manifest V3
src/
  backend/             FastAPI, ventana ILSA, sesión
  core/                Landmarks, enunciado, política de repeticiones
  classifier/          TinySkeleton + pesos + lista de clases
  semantic/            Cliente HTTP de Cloud Run (`remote.py`)
  avatar/              Cliente T-pose (`GET /tposes/{GLOSA}.pose`)
docs/ARQUITECTURA.md   Pipeline, Meet, API, captura
```

**No forma parte del runtime versionado:**

- `señario1/`, `señario2/`: láminas PNG, si las tenés en el disco.
- `extension/vendor/mediapipe/*.wasm` (y similares): regenerar con `packaging/`.
- `extension/bin/ILSA.zip` y `dist/`: salida de PyInstaller.

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
| POST | `/stt` | Modo oyente: WAV del mic de Meet → español |
| POST | `/hearing` | Modo oyente: español → glosas (y clip T-pose remoto) |
| POST | `/avatar` | Modo oyente: glosas → frames del esqueleto (servidor T-pose) |
| POST | `/activity` | “Sigo señando” (retrasa el cierre) |
| POST | `/utterance/end` | Cerrar enunciado y traducir |
| GET | `/download/exe` | `.exe` o `.bat` si existen en esta PC |

CORS: `https://meet.google.com`, localhost y orígenes `chrome-extension://`.
El API debe escuchar solo en loopback.

---

## Problemas frecuentes

| Síntoma | Qué probar |
|---------|------------|
| Extensión: “motor apagado” | Abrí ILSA, elegí modo, Encender; recargá el popup |
| ILSA arranca sin traducir | `GET` al `/health` de Cloud Run; Internet en la laptop de Meet |
| Subtítulos que no se van | Recargar extensión **y** la pestaña de Meet |
| `Extension context invalidated` | Recargaste la extensión con Meet abierto: recargá Meet |
