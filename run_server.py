"""
Servidor web LSA Meet.

    python run_server.py              # uvicorn en :8000
    python run_server.py --port 8080
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Servidor LSA Meet")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("[*] LSA Meet — arrancando servidor...", flush=True)
    print(
        "[*] La primera vez puede tardar 1–3 min (PyTorch, MediaPipe, clasificador).",
        flush=True,
    )
    print(f"[*] Puerto: {args.port}", flush=True)

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
