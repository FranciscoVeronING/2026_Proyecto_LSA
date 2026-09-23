"""
Traductor semántico LSA → español vía GGUF (llama-cpp-python o llama-server.exe).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


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
    MAX_NEW_TOKENS,
    N_CTX,
    OYENTE_GGUF_DIR,
    REPETITION_PENALTY,
    SORDO_GGUF_DIR,
    SYSTEM_PROMPT_OYENTE_PATH,
    SYSTEM_PROMPT_PATH,
    TEMPERATURE,
)

from semantic.models import (
    resolve_oyente_gguf,
    resolve_sordo_gguf,
)

SYSTEM_PROMPT = ""
SYSTEM_PROMPT_OYENTE = ""
GGUF_MODEL = None
GGUF_MODEL_OYENTE = None
_LOADED = False
_ACTIVE_MODEL_ID: Optional[str] = None
_SORDO_PATH: Optional[str] = None
_OYENTE_PATH: Optional[str] = None
_ACTIVE_CHAT_FORMAT = "chatml"
_SORDO_CHAT = "llama3"
_OYENTE_CHAT = "chatml"

_CHATML_STOP = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
_LLAMA3_STOP = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>"]

INFER_LOCK = threading.RLock()


@contextmanager
def infer_slot() -> Iterator[float]:
    """Una inferencia a la vez. Quien llega después espera en cola."""
    t0 = time.perf_counter()
    got = INFER_LOCK.acquire(blocking=False)
    if not got:
        print("[semantic] Cola: el modelo está ocupado, esperando…")
        INFER_LOCK.acquire()
        waited = time.perf_counter() - t0
        print(f"[semantic] Cola: entra a inferir tras {waited:.1f}s")
    else:
        waited = 0.0
    try:
        yield waited
    finally:
        INFER_LOCK.release()


def get_active_model_id() -> Optional[str]:
    return _ACTIVE_MODEL_ID


def loaded_paths() -> dict[str, Optional[str]]:
    return {"sordo": _SORDO_PATH, "oyente": _OYENTE_PATH}


def _read_prompt_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _load_role_prompt(folder: Path, fallback: str, default: str, beside: Optional[Path] = None) -> str:
    names = ("sys_prompt.txt", "system_prompt.txt")
    candidates: list[Path] = [folder / n for n in names]
    if beside is not None:
        parent = beside.resolve().parent
        candidates.extend(parent / n for n in names)
    candidates.append(Path(fallback))
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        text = _read_prompt_file(path)
        if text:
            print(f"[semantic] Prompt: {path}")
            return text
    return default


def load_prompt(beside: Optional[Path] = None) -> str:
    return _load_role_prompt(
        SORDO_GGUF_DIR,
        SYSTEM_PROMPT_PATH,
        "Sos un traductor estricto de Lengua de Señas Argentina (LSA) a español rioplatense natural.",
        beside=beside,
    )


def load_prompt_oyente(beside: Optional[Path] = None) -> str:
    return _load_role_prompt(
        OYENTE_GGUF_DIR,
        SYSTEM_PROMPT_OYENTE_PATH,
        "Traducí español a glosas LSA en mayúsculas, separadas por espacios. Solo la línea de glosas.",
        beside=beside,
    )


def _make_engine(model_path: str, n_threads: int, port: Optional[int] = None):
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
        from semantic.native_llama import DEFAULT_PORT, LlamaServerEngine

        return LlamaServerEngine(
            model_path=model_path,
            n_ctx=int(N_CTX),
            n_gpu_layers=0,
            n_threads=n_threads,
            port=port or DEFAULT_PORT,
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


def _chat_format_for(path: Path) -> str:
    name = path.name.lower()
    if "qwen" in name:
        return "chatml"
    return "llama3"


def _format_prompt(messages: list[dict], chat_format: Optional[str] = None) -> str:
    fmt = chat_format or _ACTIVE_CHAT_FORMAT
    if fmt == "llama3":
        return _to_llama3(messages)
    return _to_chatml(messages)


def _stop_tokens(chat_format: Optional[str] = None) -> list[str]:
    fmt = chat_format or _ACTIVE_CHAT_FORMAT
    if fmt == "llama3":
        return _LLAMA3_STOP
    return _CHATML_STOP


def unload_model() -> None:
    global GGUF_MODEL, GGUF_MODEL_OYENTE, _LOADED, _ACTIVE_MODEL_ID, _SORDO_PATH, _OYENTE_PATH

    engines = []
    for engine in (GGUF_MODEL, GGUF_MODEL_OYENTE):
        if engine is not None and engine not in engines:
            engines.append(engine)
    for engine in engines:
        try:
            if hasattr(engine, "close"):
                engine.close()
        except Exception as e:
            print(f"[semantic] Error al liberar el modelo: {e}")
    GGUF_MODEL = None
    GGUF_MODEL_OYENTE = None
    _LOADED = False
    _ACTIVE_MODEL_ID = None
    _SORDO_PATH = None
    _OYENTE_PATH = None
    gc.collect()


def load_model_and_tokenizer(model_id: Optional[str] = None, force: bool = False):
    """Carga GGUF de sordo y, si hay archivo distinto, el de oyente."""
    global SYSTEM_PROMPT, SYSTEM_PROMPT_OYENTE, GGUF_MODEL, GGUF_MODEL_OYENTE
    global _LOADED, _ACTIVE_MODEL_ID, _ACTIVE_CHAT_FORMAT, _SORDO_PATH, _OYENTE_PATH
    global _SORDO_CHAT, _OYENTE_CHAT

    del model_id
    if _LOADED and GGUF_MODEL is not None and not force:
        return

    gguf_file = resolve_sordo_gguf()
    if gguf_file is None:
        from semantic.gguf_fetch import ensure_gguf

        gguf_file = ensure_gguf()
        gguf_file = resolve_sordo_gguf() or gguf_file
    if gguf_file is None:
        raise FileNotFoundError(
            "[semantic] No está el GGUF de sordo. "
            "Poné el .gguf en src/semantic/models/sordo/"
        )

    if _LOADED or GGUF_MODEL is not None:
        print(f"[semantic] Liberando modelo {_ACTIVE_MODEL_ID}...")
        unload_model()

    oyente_file = resolve_oyente_gguf()
    SYSTEM_PROMPT = load_prompt(gguf_file)
    SYSTEM_PROMPT_OYENTE = load_prompt_oyente(oyente_file or gguf_file)
    _SORDO_CHAT = _chat_format_for(Path(gguf_file))
    _ACTIVE_CHAT_FORMAT = _SORDO_CHAT
    n_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    print(f"[semantic] Cargando GGUF sordo (CPU, {_SORDO_CHAT}): {gguf_file}")
    GGUF_MODEL = _make_engine(str(gguf_file), n_threads=n_threads)
    _SORDO_PATH = str(gguf_file)

    if oyente_file is not None and oyente_file.resolve() != Path(gguf_file).resolve():
        from semantic.native_llama import DEFAULT_PORT

        print(f"[semantic] Cargando GGUF oyente (CPU): {oyente_file}")
        GGUF_MODEL_OYENTE = _make_engine(
            str(oyente_file), n_threads=n_threads, port=DEFAULT_PORT + 1
        )
        _OYENTE_PATH = str(oyente_file)
        _OYENTE_CHAT = _chat_format_for(Path(oyente_file))
        print(f"[semantic] Oyente chat: {_OYENTE_CHAT}")
    else:
        GGUF_MODEL_OYENTE = GGUF_MODEL
        _OYENTE_PATH = _SORDO_PATH
        _OYENTE_CHAT = _SORDO_CHAT
        if oyente_file is None:
            print("[semantic] Sin GGUF en models/oyente; /oyente reusa el de sordo.")
        else:
            print("[semantic] Oyente es el mismo archivo que sordo.")

    _LOADED = True
    _ACTIVE_MODEL_ID = DEFAULT_MODEL_ID
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
    print("[semantic] Modelos listos para /sordo y /oyente.\n")


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

    with INFER_LOCK:
        raw = glosses_input.strip()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": f"Glosas: {raw}"})

        prompt = _format_prompt(messages, _SORDO_CHAT)
        print(f"[semantic] Generando traducción para: {raw!r}")
        t0 = time.perf_counter()
        response = GGUF_MODEL.create_completion(
            prompt,
            max_tokens=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            repeat_penalty=float(REPETITION_PENALTY),
            stop=_stop_tokens(_SORDO_CHAT),
        )
        text = (response["choices"][0].get("text") or "").strip()
        print(f"[semantic] Inferencia en {time.perf_counter() - t0:.1f}s → {text!r}")
        return text


def translate_spanish_to_glosses(spanish_input: str) -> str:
    """Español → una línea de glosas LSA (mismo GGUF, prompt de oyente)."""
    if not _LOADED or GGUF_MODEL is None:
        load_model_and_tokenizer()

    with INFER_LOCK:
        engine = GGUF_MODEL_OYENTE or GGUF_MODEL
        raw = spanish_input.strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_OYENTE or load_prompt_oyente()},
            {"role": "user", "content": f"Español: {raw}"},
        ]
        prompt = _format_prompt(messages, _OYENTE_CHAT)
        print(f"[semantic] Oyente → glosas para: {raw!r}")
        t0 = time.perf_counter()
        response = engine.create_completion(
            prompt,
            max_tokens=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            repeat_penalty=float(REPETITION_PENALTY),
            stop=_stop_tokens(_OYENTE_CHAT),
        )
        text = (response["choices"][0].get("text") or "").strip()
        print(f"[semantic] Oyente en {time.perf_counter() - t0:.1f}s → {text!r}")
        return text


__all__ = [
    "get_active_model_id",
    "infer_slot",
    "loaded_paths",
    "load_model_and_tokenizer",
    "switch_model",
    "translate_glosses",
    "translate_spanish_to_glosses",
    "unload_model",
]
