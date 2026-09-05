"""
Punto de entrada del proyecto.

    python run.py              # cámara + traducción
    python run.py --no-llm     # solo glosas, sin cargar la LLM
    python run.py --eval       # recorrido de evaluación del clasificador (CSV)
    python run.py --eval-semantic   # evaluación offline glosas→español (métricas + CSV)
    python run.py --probe-semantic  # probar glosas escritas, sin cámara ni clasificador

Agrega `src/` al path para que los módulos se importen igual sin importar
desde qué directorio se ejecute.

La eval semántica y el probe tienen entry point propio para no importar
OpenCV/MediaPipe antes de cargar la LLM (libera VRAM y RAM).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    if "--eval-semantic" in sys.argv:
        from app.semantic_eval import cli_main

        raise SystemExit(cli_main())

    if "--probe-semantic" in sys.argv:
        from app.semantic_probe import cli_main

        raise SystemExit(cli_main())

    from app.main import main

    main()
