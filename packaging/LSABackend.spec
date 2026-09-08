# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: genera dist/LSABackend.exe con clasificador + API."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
repo = Path(SPECPATH).resolve().parent
src = repo / "src"

datas = [
    (str(src / "classifier" / "weights"), "classifier/weights"),
    (str(src / "semantic" / "prompts"), "semantic/prompts"),
]
if (src / "semantic" / "outputs").exists():
    datas.append((str(src / "semantic" / "outputs"), "semantic/outputs"))

hidden = (
    collect_submodules("classifier")
    + collect_submodules("semantic")
    + collect_submodules("core")
    + collect_submodules("backend")
    + collect_submodules("app")
)

a = Analysis(
    [str(repo / "run_backend.py")],
    pathex=[str(src), str(repo)],
    binaries=[],
    datas=datas + collect_data_files("llama_cpp"),
    hiddenimports=hidden + ["tkinter", "tkinter.ttk", "tkinter.font"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["cv2", "mediapipe", "pyttsx3"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
