"""
Traductor semántico LSA → español vía GGUF (llama-cpp-python o llama-server.exe).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Optional


def _add_dll_dir(path: Path) -> None:
    if not path.is_dir():
        return
    path_str = str(path.resolve())
    os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path_str)
        except OSError:
            pass


def _prepare_llama_native_libs() -> None:
    """En Windows llama.dll necesita el runtime MSVC en PATH."""
    if sys.platform != "win32":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        _add_dll_dir(prefix / "Library" / "bin")
        _add_dll_dir(prefix / "bin")

    try:
        spec = importlib.util.find_spec("llama_cpp")
        origin = Path(spec.origin).resolve().parent if spec and spec.origin else None
        if origin is not None:
            _add_dll_dir(origin / "lib")
    except Exception:
        pass


_prepare_llama_native_libs()

from semantic.config import (
    DEFAULT_MODEL_ID,
    SYSTEM_PROMPT_PATH,
    MAX_NEW_TOKENS,
    N_CTX,
    TEMPERATURE,
    REPETITION_PENALTY,
)

from semantic.models import (
    list_semantic_models,
    resolve_gguf_path,
    spec_by_id,
)

SYSTEM_PROMPT = ""
GGUF_MODEL = None
_LOADED = False
_ACTIVE_MODEL_ID: Optional[str] = None
_ACTIVE_CHAT_FORMAT = "chatml"

_CHATML_STOP = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
_LLAMA3_STOP = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>"]


def get_active_model_id() -> Optional[str]:
    return _ACTIVE_MODEL_ID


def load_prompt() -> str:
    prompt_path = Path(SYSTEM_PROMPT_PATH)
    if not prompt_path.exists():
        return "Sos un traductor estricto de Lengua de Señas Argentina (LSA) a español rioplatense natural."
    return prompt_path.read_text(encoding="utf-8").strip()


def _make_engine(model_path: str, n_threads: int):
    """llama-cpp-python si está instalado; si no, llama-server.exe (Windows, CPU)."""
    try:
        from llama_cpp import Llama

        return Llama(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=int(N_CTX),
            n_batch=256,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            verbose=False,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        print(
            f"[semantic] llama-cpp-python no carga ({exc}). "
            "Uso llama-server.exe."
        )
        from semantic.native_llama import LlamaServerEngine

        return LlamaServerEngine(
            model_path=model_path,
            n_ctx=int(N_CTX),
            n_gpu_layers=0,
            n_threads=n_threads,
        )


def _to_chatml(messages: list[dict]) -> str:
    chunks = []
    for msg in messages:
        role = msg.get("role") or "user"
        content = msg.get("content") or ""
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def _to_llama3(messages: list[dict]) -> str:
    chunks = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg.get("role") or "user"
        content = msg.get("content") or ""
        chunks.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    chunks.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(chunks)


def _format_prompt(messages: list[dict]) -> str:
    if _ACTIVE_CHAT_FORMAT == "llama3":
        return _to_llama3(messages)
    return _to_chatml(messages)


def _stop_tokens() -> list[str]:
    if _ACTIVE_CHAT_FORMAT == "llama3":
        return _LLAMA3_STOP
    return _CHATML_STOP


def unload_model() -> None:
    global GGUF_MODEL, _LOADED, _ACTIVE_MODEL_ID

    if GGUF_MODEL is not None:
        try:
            if hasattr(GGUF_MODEL, "close"):
                GGUF_MODEL.close()
        except Exception as e:
            print(f"[semantic] Error al liberar el modelo: {e}")
        GGUF_MODEL = None
    _LOADED = False
    _ACTIVE_MODEL_ID = None
    gc.collect()


def load_model_and_tokenizer(model_id: Optional[str] = None, force: bool = False):
    """
    Carga el GGUF activo (llama-cpp o llama-server). Idempotente salvo ``force``.

    Args:
        model_id: Ignorado (siempre Llama 1B). Default ``llama-3.2-1b``.
        force: Recarga aunque ya esté en memoria.
    """
    global SYSTEM_PROMPT, GGUF_MODEL, _LOADED, _ACTIVE_MODEL_ID, _ACTIVE_CHAT_FORMAT

    target_id = DEFAULT_MODEL_ID
    if _LOADED and GGUF_MODEL is not None and not force and _ACTIVE_MODEL_ID == target_id:
        return

    spec = spec_by_id(target_id)
    gguf_file = resolve_gguf_path(spec)
    if gguf_file is None:
        from semantic.gguf_fetch import ensure_gguf

        gguf_file = ensure_gguf()
    if gguf_file is None:
        raise FileNotFoundError(
            f"[semantic] No está el GGUF de '{target_id}'. "
            "Abrí ILSA con internet o copiá el .gguf a %LOCALAPPDATA%\\ILSA\\models\\"
        )

    if _LOADED or GGUF_MODEL is not None:
        print(f"[semantic] Liberando modelo {_ACTIVE_MODEL_ID}...")
        unload_model()

    SYSTEM_PROMPT = load_prompt()
    _ACTIVE_CHAT_FORMAT = spec.chat_format
    n_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    print(f"[semantic] Cargando GGUF ({target_id}, CPU): {gguf_file}")
    GGUF_MODEL = _make_engine(str(gguf_file), n_threads=n_threads)

    _LOADED = True
    _ACTIVE_MODEL_ID = target_id
    print("[semantic] Calentando primera inferencia (puede tardar un poco)...")
    t0 = time.perf_counter()
    try:
        warmup_prompt = _format_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Glosas: HOLA"},
            ]
        )
        GGUF_MODEL.create_completion(
            warmup_prompt,
            max_tokens=1,
            temperature=0.0,
            stop=_stop_tokens(),
        )
    except Exception as e:
        print(f"[semantic] Warmup omitido: {e}")
    else:
        print(f"[semantic] Warmup listo en {time.perf_counter() - t0:.1f}s")
    print(f"[semantic] Modelo {target_id} listo para inferencia en tiempo real.\n")


def switch_model(model_id: str) -> str:
    """Cambia el GGUF activo. Debe llamarse desde el hilo que usa la LLM."""
    load_model_and_tokenizer(model_id=model_id, force=True)
    return _ACTIVE_MODEL_ID or model_id


def translate_glosses(glosses_input: str, history_messages: Optional[list[dict]] = None) -> str:
    """
    Glosas en una línea → oración en español (prompt + few-shot + GGUF).

    Args:
        glosses_input: Ej. ``"YO LLAMAR ESPOSO"``.
        history_messages: Turnos previos ``{role, content}`` o None.

    Returns:
        Texto generado (strip). Vacío o error → cadena vacía / lo que devuelva el engine.
    """
    if not _LOADED or GGUF_MODEL is None:
        load_model_and_tokenizer()

    raw = glosses_input.strip()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": f"Glosas: {raw}"})

    prompt = _format_prompt(messages)
    print(f"[semantic] Generando traducción para: {raw!r}")
    t0 = time.perf_counter()
    response = GGUF_MODEL.create_completion(
        prompt,
        max_tokens=int(MAX_NEW_TOKENS),
        temperature=float(TEMPERATURE),
        repeat_penalty=float(REPETITION_PENALTY),
        stop=_stop_tokens(),
    )
    text = (response["choices"][0].get("text") or "").strip()
    print(f"[semantic] Inferencia en {time.perf_counter() - t0:.1f}s → {text!r}")
    return text


__all__ = [
    "get_active_model_id",
    "list_semantic_models",
    "load_model_and_tokenizer",
    "switch_model",
    "translate_glosses",
    "unload_model",
]
