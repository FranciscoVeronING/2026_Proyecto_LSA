"""
Servidor semántico (GGUF). Local: puerto 8787. Cloud Run: variable PORT.

    python run_semantic_server.py
    python run_semantic_server.py --host 0.0.0.0 --port 8787
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main(argv=None) -> None:
    env_port = os.environ.get("PORT")
    in_cloud = bool(env_port)
    parser = argparse.ArgumentParser(description="API semántica LSA (sordo / oyente).")
    parser.add_argument(
        "--host",
        default="0.0.0.0" if in_cloud else "127.0.0.1",
        help="En contenedor / Cloud Run: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_port) if env_port else 8787,
        help="Cloud Run inyecta PORT; localmente 8787.",
    )
    args = parser.parse_args(argv)
    port = int(os.environ.get("PORT", args.port))
    host = args.host
    if in_cloud:
        host = "0.0.0.0"

    from semantic.http_server import app, warmup
    import uvicorn

    # Min-instances 0: el mmap usa CPU de arranque (--cpu-boost). El puerto
    # se abre recién cuando ready=true. ILSA reintenta /health hasta 200+ready.
    print("[semantic-http] Cargando GGUF antes de abrir el puerto…", flush=True)
    warmup()
    print(f"[semantic-http] Escuchando {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
