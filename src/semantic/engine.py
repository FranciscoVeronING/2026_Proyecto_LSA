"""One GGUF in RAM: mmap, chat template, and pinned system prompt."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


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


def prepare_windows_llama_dlls() -> None:
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


prepare_windows_llama_dlls()

from semantic.config import MAX_NEW_TOKENS, N_BATCH, N_CTX, REPETITION_PENALTY, TEMPERATURE

_CHATML_STOP = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
_LLAMA3_STOP = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>"]


def chat_format_for(path: Path) -> str:
    return "chatml" if "qwen" in path.name.lower() else "llama3"


def format_prompt(messages: list[dict], chat: str) -> str:
    if chat == "llama3":
        chunks = ["<|begin_of_text|>"]
        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content") or ""
            chunks.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        chunks.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(chunks)
    chunks = []
    for msg in messages:
        role = msg.get("role") or "user"
        content = msg.get("content") or ""
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def system_prefix(system: str, chat: str) -> str:
    if chat == "llama3":
        return (
            "<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        )
    return f"<|im_start|>system\n{system}<|im_end|>\n"


def tokenize(llm: object, text: str) -> list[int]:
    fn = getattr(llm, "tokenize", None)
    if not callable(fn):
        return []
    raw = text.encode("utf-8")
    try:
        return list(fn(raw, add_bos=False, special=True))
    except TypeError:
        return list(fn(raw))


def stop_tokens(chat: str) -> list[str]:
    return _LLAMA3_STOP if chat == "llama3" else _CHATML_STOP


def open_llama(model_path: str, n_threads: int):
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
        n_batch=N_BATCH,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        verbose=False,
        use_mmap=True,
    )
    try:
        llm = Llama(**kwargs, use_mlock=True)
    except (TypeError, RuntimeError, OSError):
        llm = Llama(**kwargs)
    print(f"[semantic] Llama() ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return llm


@dataclass
class LoadedModel:
    path: Path
    llm: object
    chat: str
    sys_n: int = 0
    sys_tokens: list[int] = field(default_factory=list)

    def pin_system(self, system: str) -> None:
        if not callable(getattr(self.llm, "eval", None)):
            return
        tokens = tokenize(self.llm, system_prefix(system, self.chat))
        if not tokens:
            return
        t0 = time.perf_counter()
        self.llm.eval(tokens)
        self.sys_tokens = tokens
        self.sys_n = len(tokens)
        print(
            f"[semantic] Sys pinned {self.path.name}: {self.sys_n} tokens "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    def rewind_system(self) -> None:
        if self.sys_n <= 0 or not hasattr(self.llm, "n_tokens"):
            return
        self.llm.n_tokens = self.sys_n
        ctx = getattr(self.llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "kv_cache_seq_rm"):
            ctx.kv_cache_seq_rm(-1, self.sys_n, -1)

    def complete(self, messages: list[dict], raw: str, route: str) -> str:
        prompt = format_prompt(messages, self.chat)
        prompt_arg: object = prompt
        extra = ""
        tokens = tokenize(self.llm, prompt)
        if tokens and self.sys_n and self.sys_tokens:
            if tokens[: self.sys_n] != self.sys_tokens:
                print(
                    f"[semantic] sys prefix mismatch {self.path.name}; "
                    f"full re-eval ({len(tokens)} tokens)",
                    flush=True,
                )
                self.sys_n = 0
            else:
                self.rewind_system()
                prompt_arg = tokens
                extra = (
                    f" prompt_tokens={len(tokens)} sys={self.sys_n} "
                    f"eval={len(tokens) - self.sys_n}"
                )
        elif tokens:
            extra = f" prompt_tokens={len(tokens)}"
        print(
            f"[semantic] /{route} with {self.path.name} for {raw!r}{extra}",
            flush=True,
        )
        t0 = time.perf_counter()
        response = self.llm.create_completion(
            prompt_arg,
            max_tokens=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            repeat_penalty=float(REPETITION_PENALTY),
            stop=stop_tokens(self.chat),
        )
        text = (response["choices"][0].get("text") or "").strip()
        if not text:
            raise RuntimeError("empty response")
        print(
            f"[semantic] {route} {self.path.name} in {time.perf_counter() - t0:.1f}s → {text!r}",
            flush=True,
        )
        return text

    def close(self) -> None:
        try:
            if hasattr(self.llm, "close"):
                self.llm.close()
        except Exception as exc:
            print(f"[semantic] Error releasing {self.path.name}: {exc}")


def load_gguf(path: Path, n_threads: int, channel: str) -> LoadedModel | None:
    resolved = path.resolve()
    print(f"[semantic] Trying {channel}: {resolved}", flush=True)
    try:
        llm = open_llama(str(resolved), n_threads=n_threads)
    except Exception as exc:
        print(f"[semantic] Init failed {path.name}: {exc}", flush=True)
        return None
    chat = chat_format_for(path)
    print(f"[semantic] GGUF in RAM ({channel}): {resolved} ({chat})", flush=True)
    return LoadedModel(path=resolved, llm=llm, chat=chat)
