"""
Arranca el motor IRIS (ventana) o solo la API.

    python run_backend.py
    python run_backend.py --headless
    python run_backend.py --no-llm
    python run_backend.py --port 8765
"""

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    sys.path.insert(0, getattr(sys, "_MEIPASS", str(Path(sys.executable).parent)))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backend.server import main

if __name__ == "__main__":
    main()
