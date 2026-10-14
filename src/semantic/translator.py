"""
Traductor semántico LSA ↔ español vía GGUF (llama-cpp-python o llama-server.exe).

Solo carga archivos en ``src/semantic/models/sordo|oyente``:
principal ``qwen2.5-1.5b.gguf``, backup ``llama-3.2-1b.gguf``.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    BACKUP_GGUF,
    MAX_NEW_TOKENS,
    N_CTX,
    OYENTE_GGUF_DIR,
    PRIMARY_GGUF,
    REPETITION_PENALTY,
    SORDO_GGUF_DIR,
    SYSTEM_PROMPT_OYENTE_PATH,
    SYSTEM_PROMPT_PATH,
    TEMPERATURE,
)
from semantic.literals import collapse_literal_runs
from semantic.models import oyente_candidates, sordo_candidates

SYSTEM_PROMPT = ""
SYSTEM_PROMPT_OYENTE = ""
_LOADED = False
_NATIVE_PORT = 18790

_CHATML_STOP = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
_LLAMA3_STOP = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>"]

INFER_LOCK = threading.RLock()


@dataclass
class LoadedEngine:
    path: Path
    engine: object
    chat: str


@dataclass
class Channel:
    name: str
    engines: list[LoadedEngine] = field(default_factory=list)
    index: int = 0

    @property
    def current(self) -> Optional[LoadedEngine]:
        if not self.engines:
            return None
        return self.engines[self.index]

    def path_str(self) -> Optional[str]:
        cur = self.current
        return str(cur.path) if cur else None

    def used_name(self) -> str:
        cur = self.current
        return cur.path.name if cur else ""


_SORDO = Channel("sordo")
_OYENTE = Channel("oyente")


@contextmanager
def infer_slot() -> Iterator[float]:
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
    return _SORDO.used_name() or None


def loaded_paths() -> dict[str, Optional[str]]:
    return {
        "sordo": _SORDO.path_str(),
        "oyente": _OYENTE.path_str(),
        "sordo_model": _SORDO.used_name() or None,
        "oyente_model": _OYENTE.used_name() or None,
    }


def _read_prompt_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _load_role_prompt(folder: Path, fallback: str, default: str) -> str:
    names = ("sys_prompt.txt", "system_prompt.txt")
    candidates = [folder / n for n in names]
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


def load_prompt() -> str:
    return _load_role_prompt(
        SORDO_GGUF_DIR,
        SYSTEM_PROMPT_PATH,
        "Sos un traductor estricto de Lengua de Señas Argentina (LSA) a español rioplatense natural.",
    )


def load_prompt_oyente() -> str:
    return _load_role_prompt(
        OYENTE_GGUF_DIR,
        SYSTEM_PROMPT_OYENTE_PATH,
        "Traducí español a glosas LSA en mayúsculas, separadas por espacios. Solo la línea de glosas.",
    )


def _make_engine(model_path: str, n_threads: int):
    global _NATIVE_PORT
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
        print(f"[semantic] llama-cpp-python no carga ({exc}). Uso llama-server.exe.")
        from semantic.native_llama import LlamaServerEngine

        port = _NATIVE_PORT
        _NATIVE_PORT += 1
        return LlamaServerEngine(
            model_path=model_path,
            n_ctx=int(N_CTX),
            n_gpu_layers=0,
            n_threads=n_threads,
            port=port,
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


def _format_prompt(messages: list[dict], chat_format: str) -> str:
    if chat_format == "llama3":
        return _to_llama3(messages)
    return _to_chatml(messages)


def _stop_tokens(chat_format: str) -> list[str]:
    if chat_format == "llama3":
        return _LLAMA3_STOP
    return _CHATML_STOP


def _close_engine(engine: object) -> None:
    try:
        if hasattr(engine, "close"):
            engine.close()
    except Exception as exc:
        print(f"[semantic] Error al liberar el modelo: {exc}")


def unload_model() -> None:
    global _LOADED, SYSTEM_PROMPT, SYSTEM_PROMPT_OYENTE
    seen = []
    for ch in (_SORDO, _OYENTE):
        for item in ch.engines:
            if item.engine not in seen:
                seen.append(item.engine)
                _close_engine(item.engine)
        ch.engines.clear()
        ch.index = 0
    SYSTEM_PROMPT = ""
    SYSTEM_PROMPT_OYENTE = ""
    _LOADED = False
    gc.collect()


def _try_init(path: Path, n_threads: int, channel: str) -> Optional[LoadedEngine]:
    print(f"[semantic] Intentando {channel}: {path}")
    try:
        engine = _make_engine(str(path), n_threads=n_threads)
    except Exception as exc:
        print(f"[semantic] Falló inicialización {path.name}: {exc}")
        return None
    chat = _chat_format_for(path)
    print(f"[semantic] GGUF en RAM ({channel}): {path.resolve()} ({chat})")
    return LoadedEngine(path=path.resolve(), engine=engine, chat=chat)


def _load_channel(channel: Channel, paths: list[Path], n_threads: int) -> None:
    if not paths:
        raise FileNotFoundError(
            f"[semantic] No hay {PRIMARY_GGUF} ni {BACKUP_GGUF} en "
            f"{'sordo' if channel.name == 'sordo' else 'oyente'}. "
            f"Esperados: {SORDO_GGUF_DIR if channel.name == 'sordo' else OYENTE_GGUF_DIR}"
        )
    loaded: list[LoadedEngine] = []
    for path in paths:
        item = _try_init(path, n_threads, channel.name)
        if item is not None:
            loaded.append(item)
            # El primero que carga es el activo (Qwen si existió y arrancó).
            if len(loaded) == 1:
                print(f"[semantic] Activo {channel.name}: {item.path}")
    if not loaded:
        names = ", ".join(p.name for p in paths)
        raise RuntimeError(f"[semantic] Ningún GGUF de {channel.name} cargó ({names}).")
    channel.engines = loaded
    channel.index = 0


def load_model_and_tokenizer(model_id: Optional[str] = None, force: bool = False):
    """Carga Qwen (principal) y Llama (backup) por canal, solo desde models/."""
    global SYSTEM_PROMPT, SYSTEM_PROMPT_OYENTE, _LOADED
    del model_id
    if _LOADED and _SORDO.current is not None and not force:
        return
    if _LOADED:
        unload_model()

    n_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    SYSTEM_PROMPT = load_prompt()
    SYSTEM_PROMPT_OYENTE = load_prompt_oyente()

    print("[semantic] Warmup: buscando GGUF locales (sin Ollama / HF / GitHub).")
    _load_channel(_SORDO, sordo_candidates(), n_threads)
    oyente_paths = oyente_candidates()
    if oyente_paths:
        _load_channel(_OYENTE, oyente_paths, n_threads)
    else:
        print(
            f"[semantic] Sin {PRIMARY_GGUF}/{BACKUP_GGUF} en {OYENTE_GGUF_DIR}; "
            "oyente usa los motores de sordo."
        )
        _OYENTE.engines = list(_SORDO.engines)
        _OYENTE.index = _SORDO.index

    _LOADED = True
    sordo = _SORDO.current
    oyente = _OYENTE.current
    print(f"[semantic] Cargado sordo → {sordo.path if sordo else '—'}")
    print(f"[semantic] Cargado oyente → {oyente.path if oyente else '—'}")
    print("[semantic] Calentando primera inferencia…")
    t0 = time.perf_counter()
    try:
        prompt = _format_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Glosas: HOLA"},
            ],
            sordo.chat if sordo else "chatml",
        )
        if sordo is None:
            raise RuntimeError("sin motor sordo")
        sordo.engine.create_completion(
            prompt, max_tokens=1, temperature=0.0, stop=_stop_tokens(sordo.chat)
        )
    except Exception as exc:
        print(f"[semantic] Warmup omitido: {exc}")
    else:
        print(f"[semantic] Warmup listo en {time.perf_counter() - t0:.1f}s")


def translate_glosses(
    glosses_input: str, history_messages: Optional[list[dict]] = None
) -> tuple[str, str]:
    if not _LOADED or _SORDO.current is None:
        load_model_and_tokenizer()

    tokens = [t for t in glosses_input.strip().split() if t]
    collapsed = collapse_literal_runs(tokens)
    raw = " ".join(collapsed)
    if raw != glosses_input.strip():
        print(f"[semantic] Deletreo/números juntados: {glosses_input.strip()!r} → {raw!r}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": f"Glosas: {raw}"})

    last_err = None
    start = _SORDO.index
    for step in range(len(_SORDO.engines)):
        idx = (start + step) % len(_SORDO.engines)
        item = _SORDO.engines[idx]
        prompt = _format_prompt(messages, item.chat)
        print(f"[semantic] /sordo con {item.path.name} para {raw!r}")
        t0 = time.perf_counter()
        try:
            response = item.engine.create_completion(
                prompt,
                max_tokens=int(MAX_NEW_TOKENS),
                temperature=float(TEMPERATURE),
                repeat_penalty=float(REPETITION_PENALTY),
                stop=_stop_tokens(item.chat),
            )
            text = (response["choices"][0].get("text") or "").strip()
            if not text:
                raise RuntimeError("respuesta vacía")
            _SORDO.index = idx
            print(
                f"[semantic] Inferencia {item.path.name} en {time.perf_counter() - t0:.1f}s → {text!r}"
            )
            return text, item.path.name
        except Exception as exc:
            last_err = exc
            print(f"[semantic] Fallo sordo {item.path.name}: {exc}")
            if item.path.name.lower() == PRIMARY_GGUF:
                print(f"[semantic] Conmutando sordo a {BACKUP_GGUF}")
    raise RuntimeError(f"[semantic] /sordo sin modelo usable: {last_err}")


def translate_spanish_to_glosses(spanish_input: str) -> tuple[str, str]:
    if not _LOADED or _OYENTE.current is None:
        load_model_and_tokenizer()

    raw = spanish_input.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_OYENTE or load_prompt_oyente()},
        {"role": "user", "content": f"Español: {raw}"},
    ]
    last_err = None
    start = _OYENTE.index
    for step in range(len(_OYENTE.engines)):
        idx = (start + step) % len(_OYENTE.engines)
        item = _OYENTE.engines[idx]
        prompt = _format_prompt(messages, item.chat)
        print(f"[semantic] /oyente con {item.path.name} para {raw!r}")
        t0 = time.perf_counter()
        try:
            response = item.engine.create_completion(
                prompt,
                max_tokens=int(MAX_NEW_TOKENS),
                temperature=float(TEMPERATURE),
                repeat_penalty=float(REPETITION_PENALTY),
                stop=_stop_tokens(item.chat),
            )
            text = (response["choices"][0].get("text") or "").strip()
            if not text:
                raise RuntimeError("respuesta vacía")
            _OYENTE.index = idx
            print(
                f"[semantic] Oyente {item.path.name} en {time.perf_counter() - t0:.1f}s → {text!r}"
            )
            return text, item.path.name
        except Exception as exc:
            last_err = exc
            print(f"[semantic] Fallo oyente {item.path.name}: {exc}")
            if item.path.name.lower() == PRIMARY_GGUF:
                print(f"[semantic] Conmutando oyente a {BACKUP_GGUF}")
    raise RuntimeError(f"[semantic] /oyente sin modelo usable: {last_err}")


__all__ = [
    "get_active_model_id",
    "infer_slot",
    "loaded_paths",
    "load_model_and_tokenizer",
    "translate_glosses",
    "translate_spanish_to_glosses",
    "unload_model",
]
