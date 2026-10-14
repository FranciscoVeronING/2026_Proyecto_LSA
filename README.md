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

Docker / Cloud Run: `Dockerfile` (puerto `$PORT` o 8787). Copiá los `.gguf` al build.
