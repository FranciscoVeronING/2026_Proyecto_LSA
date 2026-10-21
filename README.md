# Servicio semántico LSA (rama cloud)

Microservicio HTTP que traduce **glosas LSA ↔ español rioplatense** con SLM en GGUF (CPU, `llama-cpp-python`). Lo consume ILSA (edge): el clasificador local produce glosas; este servicio genera la oración (canal sordo) o las glosas para el avatar (canal oyente).

No incluye cámara, MediaPipe, ni entrenamiento. No usa Ollama, Hugging Face ni descargas en runtime: **solo archivos locales** en `src/semantic/models/`.

Inferencia **sin memoria de conversación**: cada request es system prompt + enunciado actual.

---

## Qué hace

| Canal | Dirección | Quién llama | Entrada | Salida |
| --- | --- | --- | --- | --- |
| Directo (sordo) | Glosas LSA → español | `POST /sign` | Cadena de glosas | Oración en español rioplatense |
| Inverso (oyente) | Español → glosas LSA | `POST /hearing` | Oración en español | Cadena de glosas (Topic-Comment) |
| Salud | ¿SLM en RAM? | `GET /health` | — | `ready`, rutas en disco vs en RAM |

Dos canales **independientes**: cada uno tiene su carpeta, su GGUF principal, su GGUF de backup y su `sys_prompt.txt`. No se reutiliza el motor ni el prompt entre sordo y oyente.

---

## Arquitectura (edge ↔ nube)

```
ILSA (PC)
  │  1) GET /health  hasta HTTP 200 y ready=true
  │  2) POST /sign  o  POST /hearing
  ▼
Cloud Run  (ilsa-semantic-service, us-central1)
  FastAPI + Uvicorn
  llama-cpp-python (CPU, n_gpu_layers=0)
  models/sordo/*.gguf + sys_prompt.txt
  models/oyente/*.gguf + sys_prompt.txt
```

1. El contenedor **no abre el puerto** hasta terminar el mmap del GGUF y anclar el system prompt en el KV.
2. ILSA mantiene pantalla de carga mientras `/health` falle o `ready` sea `false`.
3. Recién entonces enciende el clasificador local y envía enunciados.

Scale-to-zero: `--min-instances` 0. El cold start (1–3 min) corre con `--cpu-boost`. No hace falta CPU 24 h.

---

## Arranque del proceso

`run_semantic_server.py`:

1. Fuerza CPU: `CUDA_VISIBLE_DEVICES=""`.
2. Llama `warmup()`: carga GGUF + prompts + ancla el system prompt (`eval` del prefijo ChatML / Llama-3).
3. Recién entonces Uvicorn escucha `0.0.0.0:$PORT` (Cloud Run) o `127.0.0.1:8787` (local).

Si el warmup no deja un motor sordo en RAM, el proceso sale con error: Cloud Run no marca la instancia como lista.

Cola de inferencia: un lock (`infer_slot`). Un request a la vez por proceso; el resto espera y recibe `queued_ms`.

---

## Modelos y prompts (disco)

Nombres fijos (no se buscan otros orígenes):

| Rol | Ruta | Archivo principal | Backup |
| --- | --- | --- | --- |
| Sordo | `src/semantic/models/sordo/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |
| Oyente | `src/semantic/models/oyente/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |

En cada carpeta hace falta también `sys_prompt.txt` (el del entrenamiento de ese canal).

Orden de carga por canal: Qwen → Llama → cualquier otro `*.gguf` del folder. En RAM entra **solo el primero que inicializa** (principal). El backup queda en disco y se carga si el principal falla en una inferencia.

**Importante:** el Qwen de sordo y el de oyente son **dos GGUF distintos** (mismo nombre de archivo, pesos distintos). Copiar el mismo archivo a las dos carpetas hace que `/hearing` genere español en lugar de glosas.

Chat:

- nombre con `qwen` → plantilla ChatML, stops `<|im_end|>`, etc.
- el resto (Llama) → plantilla Llama 3, stops `<|eot_id|>`, etc.

La inferencia usa `llama-cpp-python` en CPU (`n_gpu_layers=0`).

### Canal sordo: juntado de deletreo

Antes del GGUF, `collapse_literal_runs` une letras o dígitos sueltos consecutivos:

- `NOMBRE MIO P E P A` → `NOMBRE MIO Pepa`
- `EDAD TUYO 6 7` → `EDAD TUYO 67`

El oyente no aplica esa normalización.

---

## API REST

CORS: `Allow-Origin: *`. Cuerpo JSON, UTF-8.

Token opcional: si existe `LSA_SEMANTIC_TOKEN`, hace falta `Authorization: Bearer …` o cabecera `X-LSA-Token`. Sin variable, no hay auth.

No hay historial, ni `X-LSA-Client`, ni `history` en el JSON.

### `GET /health`

Listo para ILSA **solo** con HTTP 200 y `"ready": true`. `primary_present` solo dice que el archivo está en disco.

```json
{
  "ok": true,
  "ready": true,
  "loading": false,
  "load_error": null,
  "primary": "qwen2.5-1.5b.gguf",
  "backup": "llama-3.2-1b.gguf",
  "sordo": {
    "dir": "/app/src/semantic/models/sordo",
    "primary": "qwen2.5-1.5b.gguf",
    "primary_present": true,
    "backup": "llama-3.2-1b.gguf",
    "backup_present": true
  },
  "oyente": { },
  "sordo_gguf": "/app/src/semantic/models/sordo/qwen2.5-1.5b.gguf",
  "oyente_gguf": "/app/src/semantic/models/oyente/qwen2.5-1.5b.gguf",
  "model_used_sordo": "qwen2.5-1.5b.gguf",
  "model_used_oyente": "qwen2.5-1.5b.gguf"
}
```

Durante el cold start el puerto aún no existe: el GET falla o queda colgado. Reintentar.

### `POST /sign` (canal directo)

Request:

```json
{ "glosses": "HOLA VOS COMO" }
```

`glosses`: string, 1–2000 caracteres (tokens en mayúsculas separados por espacios).

Respuesta 200:

```json
{
  "spanish": "Hola, ¿cómo estás?",
  "glosses": "HOLA VOS COMO",
  "model_used": "qwen2.5-1.5b.gguf",
  "ms": 412.3,
  "queued_ms": 0.0
}
```

| Campo | Significado |
| --- | --- |
| `spanish` | Oración generada |
| `glosses` | Eco de la entrada (ya recortada) |
| `model_used` | Nombre del GGUF que respondió |
| `ms` | Tiempo de inferencia en este request |
| `queued_ms` | Espera por el lock si había otra inferencia |

### `POST /hearing` (canal inverso)

Request:

```json
{ "spanish": "Hola, ¿cómo estás?" }
```

`spanish`: string, 1–2000 caracteres.

Respuesta 200:

```json
{
  "glosses": "HOLA VOS COMO",
  "spanish": "Hola, ¿cómo estás?",
  "model_used": "qwen2.5-1.5b.gguf",
  "ms": 380.1,
  "queued_ms": 0.0
}
```

### Errores

| HTTP | Cuándo |
| --- | --- |
| 401 | Token configurado y no coincide |
| 404 | Ruta inexistente (p. ej. `/favicon.ico`) |
| 422 | JSON inválido o campos fuera de rango |
| 503 | Sin GGUF usable, fallo de `Llama()` o respuesta vacía |

---

## Inferencia (parámetros)

| Parámetro | Valor | Notas |
| --- | --- | --- |
| `n_gpu_layers` | 0 | CPU |
| `N_CTX` | 2048 si existe `PORT` (Cloud Run); 4096 local | Override: `LSA_N_CTX` |
| `n_batch` | 512 | Prefill |
| `n_threads` | `os.cpu_count()` | En Cloud Run, 4 vCPU |
| `max_tokens` | 64 | |
| `temperature` | 0.1 | |
| `repeat_penalty` | 1.0 | |

Al cargar, se **ancla** el system prompt en el KV (`sys_n` tokens). Cada request rebobina a ese prefijo y solo evalúa glosas/oración nuevas. En logs:

```
[semantic] /sign con qwen2.5-1.5b.gguf para 'HOLA' prompt_tokens=… sys=… eval=…
```

`eval` chico = atajo del sys. Si aparece `Prefijo sys no coincide`, se reevalúa el prompt entero (latencia alta, típico ~8 s en CPU 1.5B).

---

## Variables de entorno

| Variable | Efecto |
| --- | --- |
| `PORT` | Puerto (Cloud Run). Implica `n_ctx=2048` y bind `0.0.0.0` |
| `LSA_N_CTX` | Contexto llama.cpp |
| `LSA_SEMANTIC_TOKEN` | Si está definida, exige Bearer o `X-LSA-Token` |
| `CUDA_VISIBLE_DEVICES` | El entrypoint la deja vacía |

---

## Local

```bash
pip install -r requirements.txt
# CPU llama-cpp (si no está la rueda del Dockerfile):
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

python run_semantic_server.py
# o: python run_semantic_server.py --host 0.0.0.0 --port 8787
```

Colocar los cuatro GGUF (o al menos el principal de cada canal) y los `sys_prompt.txt` en `models/sordo` y `models/oyente`.

```bash
curl -s http://127.0.0.1:8787/health
curl -s -X POST http://127.0.0.1:8787/sign -H "Content-Type: application/json" -d "{\"glosses\":\"HOLA\"}"
curl -s -X POST http://127.0.0.1:8787/hearing -H "Content-Type: application/json" -d "{\"spanish\":\"Hola\"}"
```

---

## Cloud Run

Los `*.gguf` están en `.gitignore` y **no** en `.gcloudignore`: Cloud Build sí los sube si existen en disco. `.gcloudignore` excluye `src/semantic/outputs/`, `extension/`, `dataset/`, `docs/`, etc.

```powershell
cd "D:\Mis Archivos\Proyectos\2026_Proyecto_LSA"

Copy-Item "RUTA\qwen-sordo.gguf"  "src\semantic\models\sordo\qwen2.5-1.5b.gguf" -Force
Copy-Item "RUTA\llama-sordo.gguf" "src\semantic\models\sordo\llama-3.2-1b.gguf" -Force
Copy-Item "RUTA\qwen-oyente.gguf" "src\semantic\models\oyente\qwen2.5-1.5b.gguf" -Force
Copy-Item "RUTA\llama-oyente.gguf" "src\semantic\models\oyente\llama-3.2-1b.gguf" -Force

Get-ChildItem src\semantic\models\sordo, src\semantic\models\oyente

gcloud run deploy ilsa-semantic-service `
  --source . `
  --region us-central1 `
  --memory 8Gi `
  --cpu 4 `
  --timeout 300 `
  --cpu-boost `
  --allow-unauthenticated
```

- `--timeout 300`: cold start + inferencia CPU.
- `--cpu-boost`: CPU en el arranque (mmap + ancla del sys).
- Mínimo de instancias: 0 (no usar `--min-instances 1` ni `--no-cpu-throttling` salvo costo consciente).
- Imagen: `Dockerfile` (Python 3.10, llama-cpp-python 0.3.8 CPU).

URL típica:

`https://ilsa-semantic-service-705664945052.us-central1.run.app`

---

## Contrato con ILSA

1. Poll `GET /health` hasta `ready === true` (errores de red, 502/504 y timeout = seguir en carga). Timeout de cliente largo (p. ej. 180–300 s) o reintentos cortos.
2. No prender el clasificador local antes.
3. Canal sordo: `POST /sign` con `{ "glosses": "..." }`; leer `spanish`.
4. Canal oyente: `POST /hearing` con `{ "spanish": "..." }`; leer `glosses`.
5. Ping opcional a `/health` **solo mientras la sesión está abierta** para no soltar del todo la CPU; al cerrar ILSA, cortar pings (scale to zero).

---

## Árbol relevante

```
run_semantic_server.py     entrypoint (warmup y luego Uvicorn)
Dockerfile
requirements.txt
.gcloudignore              GGUF sí se despliegan; outputs/ no
src/semantic/
  http_server.py           FastAPI
  translator.py            Llama(), pin KV, canales
  config.py                nombres, n_ctx, sampling
  models.py                candidatos GGUF
  literals.py              deletreo/dígitos (solo sordo)
  models/sordo/            GGUF + sys_prompt.txt
  models/oyente/           GGUF + sys_prompt.txt
```

Entrenamiento Unsloth, eval trial y artefactos en `src/semantic/outputs/` **no** van a la imagen Cloud Run.
