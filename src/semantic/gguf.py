"""Exportar, encontrar y correr un GGUF Q4_K_M.

En Windows, si la ruta del repo tiene espacios, Unsloth escribe el binario
en D:/temp_gguf (o en LSA_GGUF_TEMP) y después se mueve a outputs/.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from config import GGUF_STOP, MAX_NEW_TOKENS, MAX_SEQ_LENGTH
from gloss import limpiar_salida_modelo
from io_utils import log_line
from prompts import build_messages


def default_gguf_scratch_dir() -> Path:
    override = os.environ.get("LSA_GGUF_TEMP")
    if override:
        return Path(override)
    drive_d = Path("D:/temp_gguf")
    if Path("D:/").exists():
        return drive_d
    return Path(tempfile.gettempdir()) / "lsa_gguf"


def export_merged_gguf(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    log_fp: TextIO | None = None,
    scratch_dir: Path | None = None,
) -> Path:
    """Fusiona LoRA, cuantiza Q4_K_M y deja el binario en `{output_dir}_gguf/`."""
    scratch_dir = scratch_dir or default_gguf_scratch_dir()
    # Unsloth a veces tira el .gguf en scratch_gguf.
    extra_out = Path(str(scratch_dir) + "_gguf")

    for path in (scratch_dir, extra_out):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    log_line(log_fp, f"Fusionando pesos y exportando GGUF (Q4_K_M) en {scratch_dir}...")
    model.save_pretrained_gguf(str(scratch_dir), tokenizer, quantization_method="q4_k_m")

    final_dir = output_dir.parent / f"{output_dir.name}_gguf"
    final_dir.mkdir(parents=True, exist_ok=True)

    found = list(scratch_dir.glob("*.gguf")) + list(extra_out.glob("*.gguf"))
    if not found:
        raise FileNotFoundError(f"Unsloth no generó ningún .gguf en {scratch_dir} ni {extra_out}")

    for item in found:
        dest = final_dir / item.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))

    for path in (scratch_dir, extra_out):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    return final_dir


def find_q4_gguf(gguf_dir: Path) -> Path:
    candidates = list(gguf_dir.glob("*Q4_K_M.gguf")) or list(gguf_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(f"No se encontró binario GGUF en {gguf_dir}")
    return candidates[0]


def load_gguf_engine(gguf_path: Path):
    from llama_cpp import Llama

    return Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )


def predecir_glosas(llm: Any, system_prompt: str, spanish: str) -> tuple[str, float]:
    messages = build_messages(system_prompt, spanish)
    started = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        repeat_penalty=1.05,
        stop=GGUF_STOP,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = response["choices"][0]["message"]["content"].strip()
    return limpiar_salida_modelo(raw), latency_ms
