"""
Config general del proyecto.

- Compatibilidad: reexporta todo el clasificador para que
  `import config as cfg` siga funcionando en train/preprocessing/camera.
- Rutas compartidas entre módulos.
"""

import os

# ==========================================
# RUTAS COMPARTIDAS
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Reexport del clasificador (backward compatible)
from config_classifier import *  # noqa: F401,F403

# Acceso opcional al semántico sin mezclar namespaces
try:
    import config_semantic as semantic  # noqa: F401
except ImportError:
    semantic = None
