# Servicio semántico LSA

Rama **semantic-cloud**. Microservicio CPU que traduce glosas LSA ↔ español rioplatense con GGUF locales. Lo consume ILSA (edge). No hay cámara, entrenamiento ni memoria de conversación.

Cada petición es independiente: system prompt + enunciado actual.

## Canales

| Canal | HTTP | Dirección | Carpeta | GGUF principal | Backup |
| --- | --- | --- | --- | --- | --- |
| sign | `POST /sign` | glosas → español | `src/semantic/models/sign/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |
| hearing | `POST /hearing` | español → glosas | `src/semantic/models/hearing/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |

En cada carpeta va el `sys_prompt.txt` de **ese** entrenamiento. Los dos Qwen son pesos distintos con el mismo nombre de archivo: no copiar el mismo GGUF a las dos carpetas.

Qwen usa ChatML; Llama 3.2 usa plantilla Llama 3.

En RAM entra el principal de cada canal. El backup se carga solo si el principal falla en una inferencia.

`POST /sign` une letras o dígitos sueltos (`P E P A` → `Pepa`, `6 7` → `67`) antes del modelo.

## Arranque

`run_semantic_server.py` deja `CUDA_VISIBLE_DEVICES` vacío, carga ambos GGUF, ancla el system prompt en el KV y **después** abre el puerto. Cold start típico: 1–3 min. ILSA espera `GET /health` con HTTP 200 y `ready: true` antes de prender el clasificador.

Cola en proceso: un lock, una inferencia a la vez. La espera va en `queued_ms`. No hay cola entre instancias de Cloud Run.

## API

CORS `*`. JSON UTF-8. Sin historial.

Si existe `LSA_SEMANTIC_TOKEN`, hace falta `Authorization: Bearer …` o `X-LSA-Token`.

### `GET /health`

`ready` es verdadero solo con ambos motores en RAM. `primary_present` es el archivo en disco, no el mmap.

```json
{
  "ok": true,
  "ready": true,
  "loading": false,
  "load_error": null,
  "primary": "qwen2.5-1.5b.gguf",
  "backup": "llama-3.2-1b.gguf",
  "sign": { "dir": "…/models/sign", "primary_present": true, "backup_present": true },
  "hearing": { "dir": "…/models/hearing", "primary_present": true, "backup_present": true },
  "sign_gguf": "…/qwen2.5-1.5b.gguf",
  "hearing_gguf": "…/qwen2.5-1.5b.gguf",
  "model_used_sign": "qwen2.5-1.5b.gguf",
  "model_used_hearing": "qwen2.5-1.5b.gguf"
}
```

### `POST /sign`

```json
{ "glosses": "HOLA VOS COMO" }
```

```json
{
  "spanish": "Hola, ¿cómo estás?",
  "glosses": "HOLA VOS COMO",
  "model_used": "qwen2.5-1.5b.gguf",
  "ms": 412.3,
  "queued_ms": 0.0
}
```

### `POST /hearing`

```json
{ "spanish": "Hola, ¿cómo estás?" }
```

```json
{
  "glosses": "HOLA VOS COMO",
  "spanish": "Hola, ¿cómo estás?",
  "model_used": "qwen2.5-1.5b.gguf",
  "ms": 380.1,
  "queued_ms": 0.0
}
```

Errores: **401** token; **422** JSON; **503** sin GGUF o inferencia vacía. En cold start el puerto no existe: reintentar `/health`.

## Inferencia

| Parámetro | Valor |
| --- | --- |
| GPU | ninguna (`n_gpu_layers=0`) |
| `n_ctx` | 2048 si hay `PORT` (Cloud Run); 4096 local. Override: `LSA_N_CTX` |
| `n_batch` | 512 |
| hilos | `cpu_count()` |
| `max_tokens` | 64 |
| `temperature` | 0.1 |

El system prompt se evalúa una vez al arranque. Cada request rebobina a ese prefijo. En logs, `eval` chico es el atajo; `Prefijo sys no coincide` reevalúa todo (~8 s en CPU 1.5B).

## Código

```
run_semantic_server.py   warmup + Uvicorn
src/semantic/
  api.py                 FastAPI
  service.py             canales sign / hearing, cola, carga
  engine.py              llama-cpp, plantillas, KV del sys
  gguf.py                listado en disco
  config.py              rutas y sampling
  literals.py            deletreo (solo /sign)
  models/sign/           GGUF + sys_prompt.txt
  models/hearing/        GGUF + sys_prompt.txt
```

## Local

```bash
pip install -r requirements.txt
python run_semantic_server.py
curl -s http://127.0.0.1:8787/health
curl -s -X POST http://127.0.0.1:8787/sign -H "Content-Type: application/json" -d "{\"glosses\":\"HOLA\"}"
curl -s -X POST http://127.0.0.1:8787/hearing -H "Content-Type: application/json" -d "{\"spanish\":\"Hola\"}"
```

## Cloud Run

`*.gguf` está en `.gitignore`; `.gcloudignore` **sí** los manda si están en disco. Confirmar con `Get-ChildItem` antes del deploy.

```powershell
cd "D:\Mis Archivos\Proyectos\2026_Proyecto_LSA"
Copy-Item "RUTA\qwen-sign.gguf"    "src\semantic\models\sign\qwen2.5-1.5b.gguf" -Force
Copy-Item "RUTA\llama-sign.gguf"   "src\semantic\models\sign\llama-3.2-1b.gguf" -Force
Copy-Item "RUTA\qwen-hearing.gguf" "src\semantic\models\hearing\qwen2.5-1.5b.gguf" -Force
Copy-Item "RUTA\llama-hearing.gguf" "src\semantic\models\hearing\llama-3.2-1b.gguf" -Force

gcloud run deploy ilsa-semantic-service `
  --source . `
  --region us-central1 `
  --memory 8Gi `
  --cpu 4 `
  --timeout 300 `
  --cpu-boost `
  --allow-unauthenticated
```

Mínimo de instancias 0. `--cpu-boost` cubre el mmap. No dejar 4 vCPU fijas: el costo mensual es alto.

ILSA: poll `/health` hasta `ready`; luego `/sign` y `/hearing`. Ping a `/health` solo mientras la sesión está abierta.

URL típica: `https://ilsa-semantic-service-705664945052.us-central1.run.app`
