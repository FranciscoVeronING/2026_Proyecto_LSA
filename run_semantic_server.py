"""
Servidor semántico local (GGUF) para publicar con ngrok (o Cloudflare Tunnel).

    python run_semantic_server.py
    ngrok http 8787
    # o: cloudflared tunnel --url http://127.0.0.1:8787

La URL pública va en el release (packaging/upload_semantic_url.ps1).
Debug: set LSA_SEMANTIC_URL=https://xxxx.ngrok-free.dev
Token opcional: LSA_SEMANTIC_TOKEN (el mismo en ambos lados).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="API semántica LSA (sordo / oyente).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)

    from semantic.http_server import app, warmup

    warmup()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
