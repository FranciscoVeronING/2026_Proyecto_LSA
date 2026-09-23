# Semántico en tu PC + túnel HTTPS (ngrok)

El GGUF **no** corre en la nube. Corre en **esta PC** (`python run_semantic_server.py`). ngrok solo publica HTTPS hacia el puerto 8787.

## Dónde van los GGUF

- Sordo (`POST /sordo`): `src/semantic/models/sordo/*.gguf` y `sys_prompt.txt`
- Oyente (`POST /oyente`): `src/semantic/models/oyente/*.gguf` y `sys_prompt.txt`

Si falta el de oyente, se reusa el de sordo.

```bash
python run_semantic_server.py
ngrok http 8787
```

## URL pública (sin `set` en Meet)

ILSA baja sola `semantic_url.txt` del release `ilsa-llama-1b`. Cuando ngrok te dé una URL nueva:

```powershell
powershell -File packaging\upload_semantic_url.ps1 -Url https://xxxx.ngrok-free.dev
```

La PC de Meet: descomprimir ILSA.zip, abrir `ILSA.exe`, Encender. No hace falta variable de entorno.

Opcional: `LSA_SEMANTIC_URL` pisa el release (solo para debug).

Rutas: `GET /health`, `POST /sordo` `{glosses}`, `POST /oyente` `{spanish}`.
