"""
Ventana mínima del motor IRIS (Tkinter).

No es la extensión: solo enciende, apaga y reinicia FastAPI en localhost.
Los registros van a un panel discreto (útil si llama-server no arranca).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from tkinter import font as tkfont

REPO_URL = "https://github.com/FranciscoVeronING/2026_Proyecto_LSA"
AUTHORS = "Maite Nigro · Francisco Veron"

BG = "#0f1419"
BG2 = "#1a2330"
TEXT = "#e8eef6"
MUTED = "#9aa8b8"
ACCENT = "#ff9f1c"
OK = "#3ddc97"
BAD = "#ff6b6b"
WARN = "#f5c542"
LINE = "#334155"


class _QueueWriter:
    """Redirige stdout/stderr a una cola para el panel de registros."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self._orig = sys.__stdout__

    encoding = "utf-8"
    errors = "replace"
    closed = False

    def write(self, msg: str) -> int:
        if not msg:
            return 0
        self.q.put(msg)
        if self._orig:
            try:
                self._orig.write(msg)
            except Exception:
                pass
        return len(msg)

    def flush(self) -> None:
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self._orig and hasattr(self._orig, "fileno"):
            return self._orig.fileno()
        raise OSError("no fileno")


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record) + "\n")
        except Exception:
            pass


class BackendHandle:
    """Uvicorn en un hilo, con arranque / corte / reinicio."""

    def __init__(self, host: str, port: int, enable_llm: bool):
        self.host = host
        self.port = port
        self.enable_llm = enable_llm
        self.server = None
        self.thread = None
        self.starting = False

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and self.server and not self.server.should_exit)

    def _port_free(self) -> None:
        import socket

        try:
            with socket.create_connection((self.host, self.port), timeout=0.4):
                pass
        except OSError:
            return
        raise RuntimeError(
            f"El puerto {self.port} está ocupado. Cerrá otra ventana IRIS "
            "o el python run_backend.py que ya esté corriendo."
        )

    def _run_uvicorn(self) -> None:
        """Hilo de uvicorn. CUDA/Tk quedan en el hilo principal (si no, Windows se cuelga)."""
        try:
            import asyncio

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.server.run()
        except Exception as exc:
            print(f"[iris] uvicorn no arrancó: {exc}")

    def start(self) -> None:
        if self.running or self.starting:
            return
        self.starting = True
        import uvicorn
        from backend import server as srv

        try:
            self._port_free()
            os.environ.setdefault("LSA_BACKEND", "1")
            # Clasificador CUDA acá (hilo que llamó start). En IRIS debe ser el de Tk.
            srv.init_session(enable_llm=self.enable_llm)
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
                target=self._run_uvicorn, name="iris-uvicorn", daemon=True
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

    def restart(self) -> None:
        self.stop()
        time.sleep(0.25)
        self.start()


class IrisWindow:
    def __init__(self, args):
        if args.gpu:
            os.environ["LSA_USE_GPU"] = "1"

        self.args = args
        self.log_q: queue.Queue[str] = queue.Queue()
        self.log_lines: deque[str] = deque(maxlen=400)
        self._hook_logs()

        self.backend = BackendHandle(args.host, args.port, enable_llm=not args.no_llm)
        self.log_win: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None

        self.root = tk.Tk()
        self.root.title("IRIS")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self.root.after(200, self._boot)
        self.root.after(400, self._drain_logs)
        self.root.after(800, self._poll_health)

    def _hook_logs(self) -> None:
        writer = _QueueWriter(self.log_q)
        sys.stdout = writer
        sys.stderr = writer
        handler = _QueueLogHandler(self.log_q)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger("uvicorn").addHandler(handler)
        logging.getLogger("uvicorn.error").addHandler(handler)

    def _build(self) -> None:
        pad = tk.Frame(self.root, bg=BG, padx=28, pady=22)
        pad.pack(fill="both")

        title_f = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        body_f = tkfont.Font(family="Segoe UI", size=10)
        small_f = tkfont.Font(family="Segoe UI", size=9)
        tiny_f = tkfont.Font(family="Segoe UI", size=8)

        tk.Label(pad, text="IRIS", font=title_f, fg=TEXT, bg=BG).pack(anchor="w")
        tk.Label(
            pad,
            text="Intérprete de Lengua de Señas Argentina",
            font=small_f,
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            pad,
            text="Motor local para Meet: las señas se reconocen en tu PC\n"
            "y se traducen a español. El video no se sube a la nube.",
            font=body_f,
            fg=TEXT,
            bg=BG,
            justify="left",
        ).pack(anchor="w")

        card = tk.Frame(pad, bg=BG2, highlightbackground=LINE, highlightthickness=1, padx=16, pady=14)
        card.pack(fill="x", pady=(18, 12))

        row = tk.Frame(card, bg=BG2)
        row.pack(fill="x")

        self.dot = tk.Canvas(row, width=12, height=12, bg=BG2, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self._set_dot(BAD)

        self.status_var = tk.StringVar(value="Apagado")
        tk.Label(row, textvariable=self.status_var, font=body_f, fg=TEXT, bg=BG2).pack(side="left")

        btns = tk.Frame(card, bg=BG2)
        btns.pack(fill="x", pady=(12, 0))

        self.toggle_btn = tk.Button(
            btns,
            text="Encender",
            font=body_f,
            bg=ACCENT,
            fg="#111",
            activebackground="#ffb347",
            activeforeground="#111",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_toggle,
        )
        self.toggle_btn.pack(side="left")

        self.reset_btn = tk.Button(
            btns,
            text="Reiniciar",
            font=body_f,
            bg=BG2,
            fg=TEXT,
            activebackground=LINE,
            activeforeground=TEXT,
            relief="flat",
            highlightbackground=LINE,
            highlightthickness=1,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._on_reset,
            state="disabled",
        )
        self.reset_btn.pack(side="left", padx=(8, 0))

        tk.Label(pad, text=AUTHORS, font=small_f, fg=MUTED, bg=BG).pack(anchor="w", pady=(4, 0))

        link = tk.Label(
            pad,
            text="Repositorio en GitHub",
            font=small_f,
            fg=ACCENT,
            bg=BG,
            cursor="hand2",
        )
        link.pack(anchor="w")
        link.bind("<Button-1>", lambda _e: webbrowser.open(REPO_URL))

        logs = tk.Label(
            pad,
            text="Registros",
            font=tiny_f,
            fg=LINE,
            bg=BG,
            cursor="hand2",
        )
        logs.pack(anchor="e", pady=(16, 0))
        logs.bind("<Button-1>", lambda _e: self._toggle_logs())

        self.detail_var = tk.StringVar(value="")
        tk.Label(pad, textvariable=self.detail_var, font=tiny_f, fg=MUTED, bg=BG, wraplength=340, justify="left").pack(
            anchor="w", pady=(6, 0)
        )

    def _set_dot(self, color: str) -> None:
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _boot(self) -> None:
        self._busy("Encendiendo motor…")
        self.root.update_idletasks()
        self._start_on_main()

    def _start_on_main(self) -> None:
        """Arranca CUDA + API en el hilo de Tk. Un hilo extra se cuelga en Windows."""
        try:
            self.backend.start()
            self._idle("Cargando modelos…")
        except Exception as exc:
            self._idle(f"No arrancó: {exc}", error=True)

    def _on_toggle(self) -> None:
        if self.backend.running:
            self._busy("Apagando…")
            self.root.update_idletasks()
            try:
                self.backend.stop()
            except Exception:
                pass
            self._idle("Apagado")
            return
        self._busy("Encendiendo motor…")
        self.root.update_idletasks()
        self._start_on_main()

    def _on_reset(self) -> None:
        if not self.backend.running and not self.backend.starting:
            self._on_toggle()
            return
        self._busy("Reiniciando…")
        self.root.update_idletasks()
        try:
            self.backend.restart()
            self._idle("Cargando modelos…")
        except Exception as exc:
            self._idle(f"No reinició: {exc}", error=True)

    def _busy(self, label: str) -> None:
        self.status_var.set(label)
        self._set_dot(WARN)
        self.toggle_btn.configure(state="disabled")
        self.reset_btn.configure(state="disabled")

    def _idle(self, label: str, error: bool = False) -> None:
        self.status_var.set(label)
        self.toggle_btn.configure(state="normal")
        on = self.backend.running and not error
        self.reset_btn.configure(state="normal" if on else "disabled")
        self.toggle_btn.configure(text="Apagar" if on else "Encender")
        self._set_dot(OK if on else BAD)

    def _poll_health(self) -> None:
        if self.backend.running:
            try:
                import urllib.request

                with urllib.request.urlopen(
                    f"http://{self.args.host}:{self.args.port}/health", timeout=1.5
                ) as resp:
                    import json

                    data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    bits = ["En marcha"]
                    if data.get("classifier_ready"):
                        bits.append("clasificador")
                    if data.get("semantic_ready"):
                        bits.append("LLM")
                    elif self.args.no_llm:
                        bits.append("sin LLM")
                    else:
                        bits.append("LLM cargando")
                    device = data.get("device") or ""
                    self.status_var.set(" · ".join(bits))
                    self._set_dot(OK if data.get("semantic_ready") or self.args.no_llm else WARN)
                    self.toggle_btn.configure(text="Apagar", state="normal")
                    self.reset_btn.configure(state="normal")
                    extra = f"{self.args.host}:{self.args.port}"
                    if device:
                        extra += f"  ·  {device}"
                    self.detail_var.set(extra)
            except Exception:
                if self.backend.running:
                    self.status_var.set("Arrancando…")
                    self._set_dot(WARN)
        else:
            if self.status_var.get() not in ("Apagado",) and "No arrancó" not in self.status_var.get():
                if not self.toggle_btn["state"] == "disabled":
                    self.status_var.set("Apagado")
                    self._set_dot(BAD)
                    self.toggle_btn.configure(text="Encender")
                    self.reset_btn.configure(state="disabled")
                    self.detail_var.set("")
        self.root.after(1000, self._poll_health)

    def _drain_logs(self) -> None:
        try:
            while True:
                chunk = self.log_q.get_nowait()
                self.log_lines.append(chunk)
                if self.log_win is not None and self.log_text is not None:
                    self.log_text.insert("end", chunk)
                    self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(200, self._drain_logs)

    def _toggle_logs(self) -> None:
        if self.log_win is not None:
            try:
                self.log_win.destroy()
            except Exception:
                pass
            self.log_win = None
            self.log_text = None
            return
        win = tk.Toplevel(self.root)
        win.title("IRIS · registros")
        win.configure(bg=BG)
        win.geometry("520x320")
        win.protocol("WM_DELETE_WINDOW", lambda: self._toggle_logs())
        text = tk.Text(
            win,
            bg="#0b1016",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            padx=10,
            pady=10,
        )
        text.pack(fill="both", expand=True)
        text.insert("1.0", "".join(self.log_lines) or "(sin registros todavía)\n")
        text.see("end")
        self.log_win = win
        self.log_text = text

    def _on_close(self) -> None:
        try:
            self.backend.stop(join_sec=4.0)
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch(args) -> None:
    """Abre la ventana IRIS y arranca el motor en segundo plano."""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    IrisWindow(args).run()
