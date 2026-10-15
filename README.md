# Servidor semántico LSA

FastAPI: glosas → español y español → glosas. Sin Ollama ni Hugging Face:
los GGUF salen de `src/semantic/models/`.

- Sordo: `models/sordo/qwen2.5-1.5b.gguf` (backup `llama-3.2-1b.gguf`)
- Oyente: `models/oyente/qwen2.5-1.5b.gguf` (backup `llama-3.2-1b.gguf`)

```bash
pip install -r requirements.txt
python run_semantic_server.py --host 0.0.0.0 --port 8787
```

`POST /sordo` o `/translate` `{ "glosses": "..." }` → `spanish`, `model_used`  
`POST /oyente` o `/hearing` `{ "spanish": "..." }` → `glosses`, `model_used`

## Cloud Run

```powershell
cd "D:\Mis Archivos\Proyectos\2026_Proyecto_LSA"

Copy-Item "RUTA\qwen-sordo.gguf" "src\semantic\models\sordo\qwen2.5-1.5b.gguf" -Force
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

Scale-to-zero (mínimo 0). `--cpu-boost` da CPU mientras carga el GGUF **antes** de abrir el puerto (1–3 min de cold start).

`GET /health`: ILSA puede mostrar carga hasta `HTTP 200` y `"ready": true`. Mientras el contenedor arranca el GET falla o cuelga: reintentar. `primary_present` solo dice que el archivo está en disco.
