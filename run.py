"""
Punto de entrada del proyecto.

    python run.py              # cámara + traducción
    python run.py --no-llm     # solo glosas, sin cargar la LLM
    python run.py --eval       # recorrido de evaluación con salida CSV

Agrega `src/` al path para que los módulos se importen igual sin importar
desde qué directorio se ejecute.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
