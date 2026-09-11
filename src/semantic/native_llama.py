"""llama.cpp nativo (exe) para Windows sin compilar llama-cpp-python. Solo CPU."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_SEMANTIC_DIR = Path(__file__).resolve().parent
BIN_DIR = _SEMANTIC_DIR / "bin"
RELEASE = os.environ.get("LSA_LLAMA_RELEASE", "b10809")
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}"
DEFAULT_PORT = int(os.environ.get("LSA_LLAMA_PORT", "18790"))

_PACKAGES = {
    "cpu": f"llama-{RELEASE}-bin-win-cpu-x64.zip",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[semantic] Descargando {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)


def _find_server(root: Path) -> Path:
    matches = list(root.rglob("llama-server.exe"))
    if not matches:
        raise FileNotFoundError(f"No está llama-server.exe en {root}")
    return matches[0]


def ensure_llama_server(backend: str = "cpu") -> Path:
    """Instala llama-server.exe del backend pedido (cpu o vulkan)."""
    if backend not in _PACKAGES:
        backend = "cpu"
    folder = BIN_DIR / f"win-{backend}"
    try:
        return _find_server(folder)
    except FileNotFoundError:
        pass
    zip_name = _PACKAGES[backend]
    archive = BIN_DIR / zip_name
    if not archive.exists():
        _download(f"{BASE}/{zip_name}", archive)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    print(f"[semantic] Extrayendo {zip_name} ...")
    _extract_zip(archive, folder)
    return _find_server(folder)


class LlamaServerEngine:
    """Misma forma que llama_cpp.Llama.create_completion, vía HTTP local."""

    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int, n_threads: int):
        self.model_path = str(model_path)
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.port = DEFAULT_PORT
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self._log_file = None
        self.backend = "cpu"
        self._start("cpu", n_gpu_layers=0)
        atexit.register(self.close)

    def _start(self, backend: str, n_gpu_layers: int) -> None:
        """Lanza ``llama-server.exe`` en CPU y espera ``/health``."""
        exe = ensure_llama_server("cpu")
        self.backend = "cpu"
        ngl = 0
        cmd = [
            str(exe),
            "-m",
            self.model_path,
            "--port",
            str(self.port),
            "-c",
            str(self.n_ctx),
            "-ngl",
            str(ngl),
            "-t",
            str(self.n_threads),
            "--no-mmap",
        ]
        log_path = BIN_DIR / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        self._log_file = log_f
        print(f"[semantic] Arrancando llama-server en {backend.upper()} (ngl={ngl})")
        print(f"[semantic] Log: {log_path}")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(exe.parent),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def _wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server salió al arrancar. Revisá {BIN_DIR / 'llama-server.log'}"
                )
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = {}
                    status = str(payload.get("status") or "")
                    if resp.status == 200 and status.lower() in ("", "ok", "ready"):
                        print(f"[semantic] llama-server listo ({self.backend}).")
                        return
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
            except Exception as e:
                last = str(e)
            time.sleep(0.4)
        raise TimeoutError(f"llama-server no respondió en {timeout:.0f}s ({last})")

    def create_completion(self, prompt: str, max_tokens: int = 64, temperature: float = 0.1, stop=None, **kwargs):
        """
        Compatible con ``llama_cpp.Llama.create_completion``.

        Args:
            prompt: Chat ya formateado (ChatML / Llama3).
            max_tokens: ``n_predict`` del server.
            stop: Tokens de corte.

        Returns:
            ``{"choices": [{"text": str}]}``.
        """
        payload = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "temperature": float(temperature),
            "repeat_penalty": float(kwargs.get("repeat_penalty", 1.0)),
            "stop": list(stop or []),
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {e.code}: {body[:500]}") from e
        text = data.get("content") or ""
        return {"choices": [{"text": text}]}

    def close(self) -> None:
        """Mata el proceso llama-server si sigue vivo."""
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.proc = None
        log_f = self._log_file
        self._log_file = None
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
