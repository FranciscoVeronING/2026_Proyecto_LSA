# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: ILSA.exe, clasificador CPU, sin llama.cpp ni GGUF."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
repo = Path(SPECPATH).resolve().parent
src = repo / "src"

datas = [
    (str(src / "classifier" / "weights"), "classifier/weights"),
    (str(repo / "assets" / "ilsa-logo.png"), "assets"),
    (str(repo / "packaging" / "ilsa.ico"), "."),
    (str(repo / "extraction" / "__init__.py"), "extraction"),
    (str(repo / "extraction" / "meet_sequence.py"), "extraction"),
    (str(repo / "extraction" / "render_pose_pair.py"), "extraction"),
    (str(repo / "extraction" / "sign_names.py"), "extraction"),
]
try:
    _vosk_d, _vosk_b, _vosk_h = collect_all("vosk")
except Exception:
    _vosk_d, _vosk_b, _vosk_h = [], [], []
try:
    _pose_h = collect_submodules("pose_format")
except Exception:
    _pose_h = []

hidden = (
    collect_submodules("classifier")
    + collect_submodules("core")
    + collect_submodules("backend")
    + _pose_h
    + [
        "core.utterance",
        "semantic.remote",
        "semantic.config",
        "avatar.remote",
        "avatar.config",
        "avatar.names",
        "extraction.meet_sequence",
        "pose_format",
        "vosk",
        "cffi",
        "_cffi_backend",
    ]
    + list(_vosk_h)
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
    binaries=_vc_binaries + list(_vosk_b),
    datas=datas + list(_vosk_d),
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
    icon=str(repo / "packaging" / "ilsa.ico"),
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
