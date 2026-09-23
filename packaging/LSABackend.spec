# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: ILSA.exe, clasificador CPU, sin llama.cpp ni GGUF."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
repo = Path(SPECPATH).resolve().parent
src = repo / "src"

datas = [
    (str(src / "classifier" / "weights"), "classifier/weights"),
]

hidden = (
    collect_submodules("classifier")
    + collect_submodules("core")
    + collect_submodules("backend")
    + ["app.utterance", "semantic.remote", "semantic.config"]
)

_EXCLUDES = [
    "cv2",
    "mediapipe",
    "pyttsx3",
    "jax",
    "jaxlib",
    "tensorflow",
    "tensorboard",
    "keras",
    "bitsandbytes",
    "pyarrow",
    "datasets",
    "optuna",
    "matplotlib",
    "pandas",
    "sklearn",
    "scipy",
    "PIL",
    "IPython",
    "notebook",
    "triton",
    "nvidia",
    "torchaudio",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "huggingface_hub",
    "unsloth",
    "llama_cpp",
    "semantic.translator",
    "semantic.gguf_fetch",
    "semantic.native_llama",
    "semantic.http_server",
]

# VC++ al lado del exe: en una PC sin Visual Studio, shm.dll no carga (WinError 126).
import sys as _sys

_vc_binaries = []
_conda = Path(_sys.prefix) / "Library" / "bin"
if not _conda.is_dir():
    _conda = Path(_sys.prefix) / "DLLs"
for _dll in (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "VCRUNTIME140.dll",
    "VCOMP140.DLL",
):
    _p = _conda / _dll
    if _p.is_file():
        _vc_binaries.append((str(_p), "."))

a = Analysis(
    [str(repo / "run_backend.py")],
    pathex=[str(src), str(repo)],
    binaries=_vc_binaries,
    datas=datas,
    hiddenimports=hidden + ["tkinter", "tkinter.ttk", "tkinter.font", "torch.distributed"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / "rthook_cpu.py")],
    excludes=_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Torch CPU: no recortar DLL. UPX rompe shm.dll (WinError 126).
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ILSA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LSABackend",
)
