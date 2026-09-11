"""Ciclo de vida de uvicorn en un hilo (sin colgar Tk en Windows)."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time


class BackendHandle:
    """uvicorn en un hilo. start() tiene que ser el hilo de Tk."""

    def __init__(self, host: str, port: int, enable_llm: bool):
        self.host = host
        self.port = port
        self.enable_llm = enable_llm
        self.mode = "signer"
        self.semantic_model_id = ""
        self.server = None
        self.thread = None
        self.starting = False

    @property
    def running(self) -> bool:
        return bool(
            self.thread
            and self.thread.is_alive()
            and self.server
            and not self.server.should_exit
        )

    def _port_free(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.4):
                pass
        except OSError:
            return
        raise RuntimeError(
            f"El puerto {self.port} está ocupado. Cerrá otra ventana ILSA "
            "o el python run_backend.py que ya esté corriendo."
        )

    def _run_uvicorn(self) -> None:
        try:
            import asyncio

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.server.run()
        except Exception as exc:
            print(f"[ilsa] uvicorn no arrancó: {exc}")

    def start(self) -> None:
        if self.running or self.starting:
            return
        self.starting = True
        import uvicorn
        from backend import server as srv

        try:
            self._port_free()
            os.environ.setdefault("LSA_BACKEND", "1")
            llm = self.enable_llm and self.mode == "signer"
            srv.init_session(
                enable_llm=llm,
                mode=self.mode,
                semantic_model_id=self.semantic_model_id or None,
            )
            config = uvicorn.Config(
                srv.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False,
                log_config=None,
                loop="asyncio",
            )
            self.server = uvicorn.Server(config)
            self.server.install_signal_handlers = lambda: None
            self.thread = threading.Thread(
                target=self._run_uvicorn, name="ilsa-uvicorn", daemon=True
            )
            self.thread.start()
        finally:
            self.starting = False

    def stop(self, join_sec: float = 6.0) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=join_sec)
        self.thread = None
        self.server = None
        from backend import server as srv

        srv._session = None
        srv._mode = "signer"

    def restart(self) -> None:
        mode = self.mode
        model = self.semantic_model_id
        self.stop()
        time.sleep(0.25)
        self.mode = mode
        self.semantic_model_id = model
        self.start()
