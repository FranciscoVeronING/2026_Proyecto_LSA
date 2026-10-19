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
_LOADING = False
_LOAD_ERROR: Optional[str] = None
_NATIVE_PORT = 18790
_LOAD_LOCK = threading.Lock()

_CHATML_STOP = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
_LLAMA3_STOP = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>"]

INFER_LOCK = threading.RLock()


@dataclass
class LoadedEngine:
    path: Path
    engine: object
    chat: str
    sys_n: int = 0
    sys_tokens: list[int] = field(default_factory=list)


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


def load_status() -> dict:
    return {"loading": _LOADING, "error": _LOAD_ERROR}


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

        print(
            f"[semantic] Llama() mmap {model_path} n_ctx={N_CTX} threads={n_threads}…",
            flush=True,
        )
        t0 = time.perf_counter()
        kwargs = dict(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=int(N_CTX),
            n_batch=512,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            verbose=False,
            use_mmap=True,
        )
        try:
            engine = Llama(**kwargs, use_mlock=True)
        except (TypeError, RuntimeError, OSError):
            engine = Llama(**kwargs)
        print(
            f"[semantic] Llama() listo en {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return engine
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


def _system_prefix(system: str, chat_format: str) -> str:
    if chat_format == "llama3":
        return (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        )
    return f"<|im_start|>system\n{system}<|im_end|>\n"


def _tok(llm: object, text: str) -> list[int]:
    tokenize = getattr(llm, "tokenize", None)
    if not callable(tokenize):
        return []
    raw = text.encode("utf-8")
    try:
        return list(tokenize(raw, add_bos=False, special=True))
    except TypeError:
        return list(tokenize(raw))


def _pin_system(item: LoadedEngine, system: str) -> None:
    llm = item.engine
    if not callable(getattr(llm, "eval", None)):
        return
    tokens = _tok(llm, _system_prefix(system, item.chat))
    if not tokens:
        return
    t0 = time.perf_counter()
    llm.eval(tokens)
    item.sys_tokens = tokens
    item.sys_n = len(tokens)
    print(
        f"[semantic] Sys anclado {item.path.name}: {item.sys_n} tokens "
        f"en {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


def _rewind_system(item: LoadedEngine) -> None:
    if item.sys_n <= 0 or not hasattr(item.engine, "n_tokens"):
        return
    item.engine.n_tokens = item.sys_n
    ctx = getattr(item.engine, "_ctx", None)
    if ctx is not None and hasattr(ctx, "kv_cache_seq_rm"):
        ctx.kv_cache_seq_rm(-1, item.sys_n, -1)


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
    for ch in (_SORDO, _OYENTE):
        for item in ch.engines:
            _close_engine(item.engine)
        ch.engines.clear()
        ch.index = 0
    SYSTEM_PROMPT = ""
    SYSTEM_PROMPT_OYENTE = ""
    _LOADED = False
    gc.collect()


def _try_init(path: Path, n_threads: int, channel: str) -> Optional[LoadedEngine]:
    resolved = path.resolve()
    print(f"[semantic] Intentando {channel}: {resolved}", flush=True)
    try:
        engine = _make_engine(str(resolved), n_threads=n_threads)
    except Exception as exc:
        print(f"[semantic] Falló inicialización {path.name}: {exc}", flush=True)
        return None
    chat = _chat_format_for(path)
    print(f"[semantic] GGUF en RAM ({channel}): {resolved} ({chat})", flush=True)
    return LoadedEngine(path=resolved, engine=engine, chat=chat)


def _load_channel(channel: Channel, paths: list[Path], n_threads: int) -> None:
    folder = SORDO_GGUF_DIR if channel.name == "sordo" else OYENTE_GGUF_DIR
    if not paths:
        raise FileNotFoundError(
            f"[semantic] No hay {PRIMARY_GGUF} ni {BACKUP_GGUF} en {folder}."
        )
    loaded: list[LoadedEngine] = []
    for path in paths:
        item = _try_init(path, n_threads, channel.name)
        if item is not None:
            loaded.append(item)
            print(f"[semantic] Activo {channel.name}: {item.path}", flush=True)
            break
    if not loaded:
        names = ", ".join(p.name for p in paths)
        raise RuntimeError(f"[semantic] Ningún GGUF de {channel.name} cargó ({names}).")
    channel.engines = loaded
    channel.index = 0


def load_model_and_tokenizer(model_id: Optional[str] = None, force: bool = False):
    """Carga GGUF y sys_prompt de cada carpeta, sin compartir entre canales."""
    global SYSTEM_PROMPT, SYSTEM_PROMPT_OYENTE, _LOADED, _LOADING, _LOAD_ERROR
    del model_id
    with _LOAD_LOCK:
        if (
            _LOADED
            and _SORDO.current is not None
            and _OYENTE.current is not None
            and not force
        ):
            return
        if _LOADED:
            unload_model()
        _LOADING = True
        _LOAD_ERROR = None
        try:
            n_threads = max(1, os.cpu_count() or 2)
            SYSTEM_PROMPT = load_prompt()
            SYSTEM_PROMPT_OYENTE = load_prompt_oyente()

            print("[semantic] Warmup: GGUF locales por canal.", flush=True)
            for label, folder in (("sordo", SORDO_GGUF_DIR), ("oyente", OYENTE_GGUF_DIR)):
                names = []
                if folder.is_dir():
                    names = sorted(p.name for p in folder.iterdir())
                print(
                    f"[semantic] Dir {label} {folder}: {names or '(vacío o no existe)'}",
                    flush=True,
                )
            _load_channel(_SORDO, sordo_candidates(), n_threads)
            _load_channel(_OYENTE, oyente_candidates(), n_threads)

            _LOADED = True
            print(f"[semantic] Cargado sordo → {_SORDO.path_str()}", flush=True)
            print(f"[semantic] Cargado oyente → {_OYENTE.path_str()}", flush=True)
            for item in _SORDO.engines:
                _pin_system(item, SYSTEM_PROMPT)
            for item in _OYENTE.engines:
                _pin_system(item, SYSTEM_PROMPT_OYENTE)
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            raise
        finally:
            _LOADING = False


def translate_glosses(glosses_input: str) -> tuple[str, str]:
    if not _LOADED or _SORDO.current is None:
        load_model_and_tokenizer()

    tokens = [t for t in glosses_input.strip().split() if t]
    collapsed = collapse_literal_runs(tokens)
    raw = " ".join(collapsed)
    if raw != glosses_input.strip():
        print(f"[semantic] Deletreo/números juntados: {glosses_input.strip()!r} → {raw!r}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Glosas: {raw}"},
    ]

    return _complete_channel(_SORDO, messages, raw, "sordo")


def translate_spanish_to_glosses(spanish_input: str) -> tuple[str, str]:
    if not _LOADED or _OYENTE.current is None:
        load_model_and_tokenizer()

    raw = spanish_input.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_OYENTE},
        {"role": "user", "content": f"Español: {raw}"},
    ]
    return _complete_channel(_OYENTE, messages, raw, "oyente")


def _complete_channel(
    channel: Channel,
    messages: list[dict],
    raw: str,
    label: str,
) -> tuple[str, str]:
    last_err = None
    start = channel.index
    step = 0
    while step < len(channel.engines):
        idx = (start + step) % len(channel.engines)
        item = channel.engines[idx]
        step += 1
        prompt = _format_prompt(messages, item.chat)
        prompt_arg: object = prompt
        extra = ""
        tokens = _tok(item.engine, prompt)
        if tokens and item.sys_n and item.sys_tokens:
            if tokens[: item.sys_n] != item.sys_tokens:
                print(
                    f"[semantic] Prefijo sys no coincide {item.path.name}; "
                    f"re-eval completo ({len(tokens)} tokens)",
                    flush=True,
                )
                item.sys_n = 0
            else:
                _rewind_system(item)
                prompt_arg = tokens
                extra = (
                    f" prompt_tokens={len(tokens)} sys={item.sys_n} "
                    f"eval={len(tokens) - item.sys_n}"
                )
        elif tokens:
            extra = f" prompt_tokens={len(tokens)}"
        print(
            f"[semantic] /{label} con {item.path.name} para {raw!r}{extra}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            response = item.engine.create_completion(
                prompt_arg,
                max_tokens=int(MAX_NEW_TOKENS),
                temperature=float(TEMPERATURE),
                repeat_penalty=float(REPETITION_PENALTY),
                stop=_stop_tokens(item.chat),
            )
            text = (response["choices"][0].get("text") or "").strip()
            if not text:
                raise RuntimeError("respuesta vacía")
            channel.index = idx
            print(
                f"[semantic] {label} {item.path.name} en {time.perf_counter() - t0:.1f}s → {text!r}",
                flush=True,
            )
            return text, item.path.name
        except Exception as exc:
            last_err = exc
            print(f"[semantic] Fallo {label} {item.path.name}: {exc}", flush=True)
            if item.path.name.lower() == PRIMARY_GGUF:
                print(f"[semantic] Conmutando {label} a {BACKUP_GGUF}", flush=True)
                paths = sordo_candidates() if label == "sordo" else oyente_candidates()
                n_threads = max(1, os.cpu_count() or 2)
                have = {e.path.resolve() for e in channel.engines}
                system = SYSTEM_PROMPT if label == "sordo" else SYSTEM_PROMPT_OYENTE
                for path in paths:
                    if path.resolve() in have:
                        continue
                    nxt = _try_init(path, n_threads, channel.name)
                    if nxt is None:
                        continue
                    _pin_system(nxt, system)
                    channel.engines.append(nxt)
                    break
    raise RuntimeError(f"[semantic] /{label} sin modelo usable: {last_err}")


__all__ = [
    "get_active_model_id",
    "infer_slot",
    "loaded_paths",
    "load_status",
    "load_model_and_tokenizer",
    "translate_glosses",
    "translate_spanish_to_glosses",
    "unload_model",
]
