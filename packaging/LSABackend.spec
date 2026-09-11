# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: dist/LSABackend/ILSA.exe. CPU only, sin GGUF (se descarga al abrir)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
repo = Path(SPECPATH).resolve().parent
src = repo / "src"

datas = [
    (str(src / "classifier" / "weights"), "classifier/weights"),
    (str(src / "semantic" / "prompts"), "semantic/prompts"),
]

hidden = (
    collect_submodules("classifier")
    + collect_submodules("semantic")
    + collect_submodules("core")
    + collect_submodules("backend")
    + ["app.utterance"]
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
    "torch.distributed",
    "torchaudio",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "huggingface_hub",
    "unsloth",
]

a = Analysis(
    [str(repo / "run_backend.py")],
    pathex=[str(src), str(repo)],
    binaries=[],
    datas=datas + collect_data_files("llama_cpp"),
    hiddenimports=hidden + ["tkinter", "tkinter.ttk", "tkinter.font"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / "rthook_cpu.py")],
    excludes=_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

_GPU_MARK = (
    "cublas",
    "cudart",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvrtc",
    "nvjitlink",
    "nvtx",
    "npp",
    "nccl",
    "torch_cuda",
    "c10_cuda",
    "libtorch_cuda",
    "ggml-cuda",
    "cublaslt",
    "nvtools",
    "nvidia",
    "cusparselt",
)


def _cpu_only(items):
    kept = []
    for item in items:
        name = item[0] if isinstance(item, (tuple, list)) else str(item)
        low = str(name).replace("\\", "/").lower()
        if any(mark in low for mark in _GPU_MARK):
            continue
        if "/nvidia/" in low:
            continue
        kept.append(item)
    return kept


a.binaries = _cpu_only(a.binaries)
a.datas = _cpu_only(a.datas)

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
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LSABackend",
)
