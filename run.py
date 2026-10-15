"""
Punto de entrada del proyecto.

    python run.py              # cámara + traducción
    python run.py --no-llm     # solo glosas, sin llamar al traductor remoto
    python run.py --eval       # recorrido de evaluación del clasificador (CSV)

Agrega `src/` al path para que los módulos se importen igual sin importar
desde qué directorio se ejecute.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from app.main import main

    main()
