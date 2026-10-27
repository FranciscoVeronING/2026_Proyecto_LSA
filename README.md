# LSA semantic service

Branch **semantic-cloud**. CPU microservice that translates LSA glosses ↔ Rioplatense Spanish with local GGUF files. ILSA (edge) is the client. No camera, training, or conversation memory.

Each request is independent: system prompt + current utterance.

Model **system prompts** (`sys_prompt.txt`) and the user prefixes `Glosas:` / `Español:` stay in Spanish. That is the training contract of the SLMs, not UI copy.

## Channels

| Channel | HTTP | Direction | Folder | Primary GGUF | Backup |
| --- | --- | --- | --- | --- | --- |
| sign | `POST /sign` | glosses → Spanish | `src/semantic/models/sign/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |
| hearing | `POST /hearing` | Spanish → glosses | `src/semantic/models/hearing/` | `qwen2.5-1.5b.gguf` | `llama-3.2-1b.gguf` |

Each folder needs the `sys_prompt.txt` from **that** training run. The two Qwen files share a filename and **not** the same weights. Do not copy one GGUF into both folders.

Qwen uses ChatML; Llama 3.2 uses the Llama 3 template.

RAM holds the primary model per channel. The backup loads only if the primary fails at inference time.

`POST /sign` joins consecutive single letters or digits (`P E P A` → `Pepa`, `6 7` → `67`) before the model.

## Startup

`run_semantic_server.py` clears `CUDA_VISIBLE_DEVICES`, loads both GGUFs, pins the system prompt in the KV cache, then opens the port. Typical cold start: 1–3 min. ILSA waits for `GET /health` with HTTP 200 and `ready: true` before enabling the local classifier.

In-process queue: one lock, one inference at a time. Wait time is `queued_ms`. There is no shared queue across Cloud Run instances.

## API

CORS `*`. UTF-8 JSON. No history.

If `LSA_SEMANTIC_TOKEN` is set, send `Authorization: Bearer …` or `X-LSA-Token`.

### `GET /health`

`ready` is true only when both engines are in RAM. `primary_present` means the file is on disk, not that it is mmap'd.

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

Errors: **401** bad token; **422** invalid JSON; **503** missing GGUF or empty generation. During cold start the port is not open: retry `/health`.

## Inference

| Parameter | Value |
| --- | --- |
| GPU | none (`n_gpu_layers=0`) |
| `n_ctx` | 2048 if `PORT` is set (Cloud Run); 4096 locally. Override: `LSA_N_CTX` |
| `n_batch` | 512 |
| threads | `cpu_count()` |
| `max_tokens` | 64 |
| `temperature` | 0.1 |

The system prompt is evaluated once at startup. Each request rewinds to that prefix. In logs, a small `eval` is the fast path; `sys prefix mismatch` re-evaluates the full prompt (~8 s on CPU 1.5B).

## Code layout

```
run_semantic_server.py   warmup + Uvicorn
src/semantic/
  api.py                 FastAPI
  service.py             sign / hearing channels, queue, load
  engine.py              llama-cpp, chat templates, sys KV
  gguf.py                on-disk inventory
  config.py              paths and sampling
  literals.py            letter/digit collapse (/sign only)
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

`*.gguf` is in `.gitignore`; `.gcloudignore` still ships them if they exist on disk. Check with `Get-ChildItem` before deploy.

```powershell
cd "D:\Mis Archivos\Proyectos\2026_Proyecto_LSA"
Copy-Item "PATH\qwen-sign.gguf"     "src\semantic\models\sign\qwen2.5-1.5b.gguf" -Force
Copy-Item "PATH\llama-sign.gguf"    "src\semantic\models\sign\llama-3.2-1b.gguf" -Force
Copy-Item "PATH\qwen-hearing.gguf"  "src\semantic\models\hearing\qwen2.5-1.5b.gguf" -Force
Copy-Item "PATH\llama-hearing.gguf" "src\semantic\models\hearing\llama-3.2-1b.gguf" -Force

gcloud run deploy ilsa-semantic-service `
  --source . `
  --region us-central1 `
  --memory 8Gi `
  --cpu 4 `
  --timeout 300 `
  --cpu-boost `
  --allow-unauthenticated
```

Min instances: 0. `--cpu-boost` covers mmap. Do not keep 4 vCPUs allocated idle; monthly cost is high.

ILSA: poll `/health` until `ready`, then `/sign` and `/hearing`. Optional `/health` ping only while a session is open.

Typical URL: `https://ilsa-semantic-service-705664945052.us-central1.run.app`
