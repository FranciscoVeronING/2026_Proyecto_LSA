"""
Entrypoint. Local: 127.0.0.1:8787. Cloud Run: 0.0.0.0 and PORT.

    python run_semantic_server.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from semantic.config import LOCAL_PORT


def main(argv=None) -> None:
    env_port = os.environ.get("PORT")
    in_cloud = bool(env_port)
    parser = argparse.ArgumentParser(description="LSA semantic API (/sign, /hearing).")
    parser.add_argument("--host", default="0.0.0.0" if in_cloud else "127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_port) if env_port else LOCAL_PORT,
    )
    args = parser.parse_args(argv)
    port = int(os.environ.get("PORT", args.port))
    host = "0.0.0.0" if in_cloud else args.host

    from semantic.api import app, warmup
    import uvicorn

    print("[semantic] Loading GGUF before opening the port…", flush=True)
    warmup()
    print(f"[semantic] Listening {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
