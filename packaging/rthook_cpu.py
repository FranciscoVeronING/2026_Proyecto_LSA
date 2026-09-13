# PyInstaller: el motor es 100% CPU (Torch CPU + llama.cpp CPU).
# Llama 1B no usa placa de video.
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.pop("LSA_USE_GPU", None)

if getattr(sys, "frozen", False):
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    exe_dir = Path(sys.executable).resolve().parent
    roots.append(exe_dir)
    roots.append(exe_dir / "_internal")
    extra = []
    for root in roots:
        extra.extend(
            [
                root,
                root / "torch" / "lib",
                root / "torch" / "bin",
                root / "llama_cpp" / "lib",
            ]
        )
    seen = set()
    for folder in extra:
        key = str(folder.resolve()) if folder.exists() else ""
        if not key or key in seen:
            continue
        seen.add(key)
        os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(key)
            except OSError:
                pass
