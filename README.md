# Intérprete de Lengua de Señas Argentina (ILSA)

Este repositorio es el **cliente de Google Meet**: una extensión Chrome más un
motor local (`ILSA.exe` o `python run_backend.py`). Traduce **LSA ↔ español**
en una videollamada. El video no sale de la PC. A la nube van glosas o
español, nunca JPEG.

Leé este archivo primero. El detalle de cada pieza está en `docs/`.

| Documento | Qué cubre |
|-----------|-----------|
| [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) | Cómo funciona el producto de punta a punta, API, clasificador, Cloud Run |
| [docs/EXTENSION.md](docs/EXTENSION.md) | Extensión MV3, Meet, Holistic, mute, canvas, HUD |
| [docs/AVATAR.md](docs/AVATAR.md) | Esqueleto del oyente: T-pose remoto, stems, compositor, canvas 50/50 |
| [docs/DICCIONARIO.md](docs/DICCIONARIO.md) | Laboratorio que **fabrica** los `.pose` (PDF, QR, YouTube). No corre en Meet |
| [docs/archivo/entregable_extension_2026-09.md](docs/archivo/entregable_extension_2026-09.md) | Ficha histórica (septiembre 2026). **No** describe el runtime actual |

---

## Qué es y qué no es

**Es** un intérprete en vivo para Meet:

- **Modo sordo.** Quien seña. MediaPipe saca un esqueleto. TinySkeleton nombra
  cada seña (glosa). Tras ~4 s de pausa, Cloud Run arma una oración en español.
  Esa oración se pinta abajo del video que Meet ya envía.
- **Modo oyente.** Quien habla o escribe en el chat. Vosk transcribe el mic de
  Meet (o se usa el texto del chat). Cloud Run devuelve glosas. El canvas se
  parte 50/50: cámara a la izquierda, esqueleto T-pose a la derecha.

**No es** un entrenador de redes, un diccionario pedagógico ni un servidor GGUF.
Los pesos `.pth` se asumen ya exportados. El Llama vive en la rama
`semantic-cloud` y en Cloud Run. Los `.pose` del diccionario de movimiento
viven en otro Cloud Run (`tpose-server`). Carpetas tipo `señario1/` con PNG, si
están en el disco, **el código no las usa**.

---

## Tres procesos

```
┌─ Chrome (extensión `extension/`) ─────────────────────────────┐
│  Meet: cámara, mic, chat, canvas, Holistic WASM, HUD          │
└───────────────────────────┬───────────────────────────────────┘
                            │ HTTP loopback
                            ▼
┌─ ILSA  127.0.0.1:8765  (`run_backend.py` / ILSA.exe) ─────────┐
│  Sordo: TinySkeleton CPU                                      │
│  Oyente: Vosk (WAV → español) + compositor de .pose           │
│  Ambos: clientes HTTP de Cloud Run                            │
└──────┬────────────────────────────────────────────┬───────────┘
       │ POST /sign  POST /hearing                  │ GET /tposes/{GLOSA}.pose
       ▼                                            ▼
  ilsa-semantic-service                    tpose-server
  us-central1                              southamerica-east1
```

La extensión **no** carga PyTorch ni GGUF. ILSA **no** corre MediaPipe. Meet
nunca recibe el stream crudo de la cámara cuando ILSA está encendido: recibe un
`canvas.captureStream`.

---

## Qué necesitás

- Windows (el exe se arma con PyInstaller en esta PC).
- Python 3.9–3.12 para desarrollar. El esqueleto oyente pide el paquete
  `pose-format` (está en `requirements.txt`; ILSA lo instala si falta).
- Chrome, para la extensión.
- Internet en la laptop de Meet (semántico + T-pose).
- Los pesos (`src/classifier/weights/`) suelen ir con **Git LFS**:

```bash
git lfs pull
```

No hace falta GGUF en esta PC. Las URLs de Cloud Run están en
`src/semantic/config.py` y `src/avatar/config.py`. Overrides opcionales:
`LSA_SEMANTIC_URL`, `LSA_AVATAR_URL`, o el archivo
`%LOCALAPPDATA%\ILSA\avatar_url.txt`.

---

## Instalación de Python

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` es el runtime de ILSA (Torch CPU, FastAPI, Vosk). MediaPipe
corre en Chrome, no en este `pip`.

---

## Cómo correrlo

### 1. MediaPipe de la extensión (una vez)

Los WASM/tflite de Holistic no se commitean.

```bash
py -3 packaging/fetch_extension_assets.py
```

Eso llena `extension/vendor/mediapipe/` y, si faltan, los íconos.

### 2. Motor ILSA

**Desarrollo:**

```bash
python run_backend.py
```

Ventana Tk: splash (ping a Cloud Run), Sordo / Oyente, Encender. Dejala abierta.
Opciones: `--headless`, `--mode signer|hearing`, `--port 8765`, `--no-llm`.

**Quien usa el exe:** en el popup de la extensión, **Instalar motor → Descargar
ILSA**. Descomprimí, abrí **ILSA.exe**. Elegí modo y Encender.

**Quien publica el zip** (en esta PC):

```bash
packaging\build_exe.bat
powershell -File packaging\upload_ilsa_zip.ps1
```

Hace falta GitHub CLI autenticado (`gh auth login`) para subir el zip. El zip
es clasificador + Vosk + cliente HTTP. El GGUF no viaja.

### 3. Cargar la extensión

1. Chrome → `chrome://extensions`
2. Activar **Modo de desarrollador**
3. **Cargar descomprimida** → carpeta `extension/`
4. Tras cambiar `manifest.json` o content scripts, **recargar la extensión**.
   Tras cambiar el hook de cámara, **recargar la pestaña de Meet**.

Al instalar se abre `welcome.html`. El popup muestra si el motor está
conectado. No enciende ILSA: eso es la ventana del exe.

### 4. Google Meet

1. ILSA abierto y Encendido.
2. Entrá a `https://meet.google.com/...` y permití la cámara.
3. Con el motor sano, la traducción arranca sola. **Apagá** en ILSA para pararla.

**Sordo:** Holistic → `POST /sign` → pausa 4 s → español quemado 8 s (el reloj
se reinicia si llega otra oración). Mute o cámara apagada cortan captura.

**Oyente:** mic de Meet y/o chat → glosas en el video + esqueleto a la derecha.
Mute corta el STT. Cámara apagada no clasifica (modo sordo) y no alimenta el
canvas.

Detalle de Meet: [docs/EXTENSION.md](docs/EXTENSION.md). Esqueleto:
[docs/AVATAR.md](docs/AVATAR.md).

---

## Cómo se decide “una seña” (modo sordo)

Alineado entre `src/classifier/config.py` y `extension/lib/capture.js`:

1. **Empieza** cuando hay al menos una mano usable (modo `auto`).
2. **Termina** cuando el buffer llega a 60 frames **o** ambas manos desaparecen
   ~0,4 s (se repite el último frame en esa gracia). No se corta por manos
   quietas: hay señas estáticas.
3. Se recorta o submuestrea a **16 frames** y se manda al clasificador.
4. Si la confianza es baja, hay cooldown o la seña es demasiado corta, se
   descarta.
5. Tras ~**4 s** sin actividad de señado se cierra el enunciado y corre el
   traductor remoto.

El clasificador espera `(1, 16, 225)`: 16 frames × (33×3 pose + 21×3 + 21×3
manos). La cara de Holistic **no** entra a ese vector. El video **nunca** entra
a PyTorch.

---

## Mapa del repositorio

```
run_backend.py              Arranque: ventana ILSA o --headless
LSABackend.bat              Atajo Windows
requirements.txt            pip del motor (sin MediaPipe)
assets/ilsa-logo.png        Logo de la ventana y de la extensión
packaging/                  WASM MediaPipe, PyInstaller, íconos, subida del zip
extension/                  Extensión Manifest V3 (Meet)
  manifest.json
  background.js             Service worker
  popup.html / popup.js
  welcome.html
  offscreen.html / .js      Holistic + recorte de seña (sin pestaña)
  sandbox.html / .js        WASM (CSP relajado)
  content/inject-gum.js     getUserMedia, canvas, mic, mute
  content/meet.js           HUD, JPEG, STT/hearing, chat
  content/skeleton.js       Dibujo del esqueleto oyente
  lib/                      api, capture, landmarks, prefs
  vendor/mediapipe/         gitignored; fetch_extension_assets.py
  bin/                      ILSA.zip de publicación (no versionar)
src/
  backend/                  FastAPI, Tk, sesión sordo/oyente, Vosk
  classifier/               TinySkeleton + pesos + clases
  core/                     Landmarks, enunciado, repeticiones
  semantic/                 Cliente Cloud Run (runtime: remote.py + config.py)
  avatar/                   Cliente T-pose (stems, GET, caché, compositor)
extraction/                 Laboratorio de .pose + compositor en vivo
  meet_sequence.py          .pose bajados → frames del canvas
  render_pose_pair.py       Trim, park, EMA, puentes (también mp4 de prueba)
docs/                       Arquitectura, extensión, avatar, diccionario
```

**No forma parte del runtime versionado:**

- `señario1/`, `señario2/`: láminas, si están en el disco.
- `extension/vendor/mediapipe/*.wasm`: regenerar con `packaging/`.
- `extension/bin/ILSA.zip` y `dist/`: salida de PyInstaller.
- `extraction/out/poses/`: diccionario local de laboratorio (~672 MB, gitignore).
  Meet **no** lo lee; pide T-pose a Cloud Run.
- Caché de usuario: `%LOCALAPPDATA%\ILSA\` (Vosk, T-pose, `client_id.txt`).

En `src/semantic/` pueden quedar scripts viejos de entrenamiento/GGUF. El
**runtime de esta rama** solo usa `remote.py` y `config.py`. El servidor Llama
está en la rama `semantic-cloud`.

---

## API del motor (`127.0.0.1:8765`)

CORS: `https://meet.google.com`, localhost y orígenes `chrome-extension://`.
El API tiene que escuchar en loopback. Contrato y JSON:
[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

| Método | Ruta | Uso |
|--------|------|-----|
| GET | `/health` | Motor, modo, clasificador, traductor |
| GET | `/metrics` | Tiempos y conteos de la sesión |
| GET | `/config` | Umbrales de captura |
| GET | `/state` | Glosas pendientes y último texto |
| POST | `/session` | Nueva sesión (`left_handed`) |
| POST | `/sign` | Frames de una seña (solo sordo) |
| POST | `/stt` | WAV del mic de Meet → español (oyente) |
| POST | `/hearing` | Español → glosas + clip T-pose (oyente) |
| POST | `/activity` | Retrasa el cierre del enunciado |
| POST | `/utterance/end` | Cerrar enunciado y traducir |
| GET | `/mediapipe/{name}` | Sirve WASM locales a la extensión |
| GET | `/download/exe` | `.exe` o `.bat` si existen en esta PC |

---

## Servicios remotos

| Rol | URL (default) | Rama / código |
|-----|---------------|----------------|
| Español ↔ glosas | `https://ilsa-semantic-service-705664945052.us-central1.run.app` | `semantic-cloud` |
| Clips `.pose` | `https://tpose-server-705664945052.southamerica-east1.run.app` | `avatar-cloud` |

Sordo: `POST /sign` `{ "glosses" }` → `{ "spanish" }`.  
Oyente: `POST /hearing` `{ "spanish" }` → `{ "glosses" }`.  
(Esos paths son del **Cloud Run**. En ILSA local, `POST /sign` son landmarks
y `POST /hearing` es español → glosas + clip.)  
T-pose: `GET /tposes/{STEM}.pose` (igualdad de nombre; ver
[docs/AVATAR.md](docs/AVATAR.md)).

---

## Problemas frecuentes

| Síntoma | Qué probar |
|---------|------------|
| Extensión: “motor apagado” | ILSA abierto, modo elegido, Encender; recargar el popup |
| ILSA arranca sin traducir | `GET` al `/health` de Cloud Run; Internet en esta PC |
| Subtítulos que no se van | Recargar extensión **y** la pestaña de Meet |
| `Extension context invalidated` | Recargaste la extensión con Meet abierto: recargá Meet |
| Cámara entera, sin esqueleto | Modo oyente + esta carpeta `extension/` |
| Esqueleto quieto / “sin clip” | T-pose Cloud Run, stem de la glosa, `LSA_AVATAR_URL` |
| Mute y sigue el STT | Recargar Meet; el tap mira el botón de mic de Meet |
| Cámara apagada y sigue clasificando | Recargar Meet; el hook mira el botón de cámara |
