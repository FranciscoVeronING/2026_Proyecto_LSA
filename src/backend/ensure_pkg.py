"""If a module is missing in development, install it with ILSA's Python."""

from __future__ import annotations

import importlib
import subprocess
import sys


def ensure_module(import_name: str, pip_spec: str) -> None:
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        pass
    if getattr(sys, "frozen", False):
        raise ImportError(
            f"No module named {import_name!r}. Rebuild ILSA.exe with that library."
        )
    print(f"[ilsa] pip install {pip_spec}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pip_spec],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    importlib.invalidate_caches()
    importlib.import_module(import_name)
