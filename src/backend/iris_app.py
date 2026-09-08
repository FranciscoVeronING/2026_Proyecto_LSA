"""
Ventana mínima del motor ILSA (Tkinter).

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

BG = "#1D3E53"
BG2 = "#26516D"
TEXT = "#FFFFFF"
MUTED = "#EAF4FA"
ACCENT = "#5BCBE8"
ACCENT_FG = "#1D3E53"
BTN2 = "#1D3E53"
BTN2_FG = "#FFFFFF"
OK = "#5BCBE8"
BAD = "#E89A94"
WARN = "#5BCBE8"
LINE = "#1A4A63"
LOG_BG = "#163040"
DISABLED_BG = "#152A38"
DISABLED_FG = "#7A9BB0"


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
        self.mode = "signer"
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
            f"El puerto {self.port} está ocupado. Cerrá otra ventana ILSA "
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
            srv.init_session(enable_llm=llm, mode=self.mode)
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
        self.stop()
        time.sleep(0.25)
        self.mode = mode
        self.start()


class IlsaWindow:
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
        self.root.title("ILSA")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
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

        tk.Label(pad, text="ILSA", font=title_f, fg=TEXT, bg=BG).pack(anchor="w")
        tk.Label(
            pad,
            text="Intérprete de Lengua de Señas Argentina",
            font=small_f,
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            pad,
            text="Elegí el modo y después encendé el motor.\n"
            "Sordo: señas LSA → español. Oyente: tu voz → subtítulos en cámara.",
            font=body_f,
            fg=TEXT,
            bg=BG,
            justify="left",
        ).pack(anchor="w")

        modes = tk.Frame(pad, bg=BG)
        modes.pack(fill="x", pady=(14, 0))
        self.mode_var = tk.StringVar(value="")

        def mode_row(parent, value, title, subtitle):
            wrap = tk.Frame(parent, bg=BG2, highlightbackground=LINE, highlightthickness=1)
            wrap.pack(fill="x", pady=(0, 8))
            inner = tk.Frame(wrap, bg=BG2, padx=12, pady=10)
            inner.pack(fill="x")
            tk.Radiobutton(
                inner,
                text=title,
                variable=self.mode_var,
                value=value,
                font=body_f,
                fg=TEXT,
                bg=BG2,
                activebackground=BG2,
                activeforeground=TEXT,
                selectcolor=BG,
                highlightthickness=0,
                command=self._on_mode_change,
            ).pack(anchor="w")
            tk.Label(inner, text=subtitle, font=tiny_f, fg=MUTED, bg=BG2, justify="left").pack(
                anchor="w", padx=(22, 0)
            )
            return wrap

        self.mode_signer = mode_row(
            modes,
            "signer",
            "Sordo · LSA → español",
            "La cámara lee las señas y las traduce. Es el flujo actual.",
        )
        self.mode_hearing = mode_row(
            modes,
            "hearing",
            "Oyente · voz → subtítulos",
            "Transcribe lo que decís y lo pinta en tu video de Meet.\n"
            "Más adelante: audio → glosas LSA.",
        )

        card = tk.Frame(pad, bg=BG2, highlightbackground=LINE, highlightthickness=1, padx=16, pady=14)
        card.pack(fill="x", pady=(18, 12))

        row = tk.Frame(card, bg=BG2)
        row.pack(fill="x")

        self.dot = tk.Canvas(row, width=12, height=12, bg=BG2, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self._set_dot(BAD)

        self.status_var = tk.StringVar(value="Elegí un modo")
        tk.Label(row, textvariable=self.status_var, font=body_f, fg=TEXT, bg=BG2).pack(side="left")

        btns = tk.Frame(card, bg=BG2)
        btns.pack(fill="x", pady=(12, 0))

        self.toggle_btn = tk.Button(
            btns,
            text="Encender",
            font=body_f,
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            command=self._on_toggle,
        )
        self.toggle_btn.pack(side="left")

        self.reset_btn = tk.Button(
            btns,
            text="Reiniciar",
            font=body_f,
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            command=self._on_reset,
        )
        self.reset_btn.pack(side="left", padx=(8, 0))

        self._toggle_armed = False
        self._reset_armed = False
        self._sync_actions()

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
            fg=MUTED,
            bg=BG,
            cursor="hand2",
        )
        logs.pack(anchor="e", pady=(16, 0))
        logs.bind("<Button-1>", lambda _e: self._toggle_logs())

        self.detail_var = tk.StringVar(value="")
        tk.Label(pad, textvariable=self.detail_var, font=tiny_f, fg=MUTED, bg=BG, wraplength=340, justify="left").pack(
            anchor="w", pady=(6, 0)
        )

    def _on_mode_change(self) -> None:
        if self.backend.running:
            return
        mode = self.mode_var.get()
        if mode == "signer":
            self.status_var.set("Listo · modo sordo")
        elif mode == "hearing":
            self.status_var.set("Listo · modo oyente")
        self._sync_actions()

    def _paint_btn(self, btn: tk.Button, *, live: bool, role: str) -> None:
        """live=se puede clickear. Windows ignora colores si state=disabled."""
        if not live:
            btn.configure(
                state="normal",
                bg=DISABLED_BG,
                fg=DISABLED_FG,
                activebackground=DISABLED_BG,
                activeforeground=DISABLED_FG,
                relief="flat",
                bd=0,
                highlightthickness=0,
                cursor="arrow",
            )
            return
        if role == "start":
            btn.configure(
                bg=ACCENT,
                fg=ACCENT_FG,
                activebackground="#7AD7EE",
                activeforeground=ACCENT_FG,
                relief="raised",
                bd=3,
                highlightthickness=0,
                cursor="hand2",
            )
        elif role == "stop":
            btn.configure(
                bg=TEXT,
                fg=ACCENT_FG,
                activebackground="#EAF4FA",
                activeforeground=ACCENT_FG,
                relief="raised",
                bd=3,
                highlightthickness=0,
                cursor="hand2",
            )
        else:
            btn.configure(
                bg=BTN2,
                fg=TEXT,
                activebackground="#1A4A63",
                activeforeground=TEXT,
                relief="raised",
                bd=3,
                highlightbackground=ACCENT,
                highlightthickness=1,
                cursor="hand2",
            )

    def _sync_actions(self, busy: bool = False) -> None:
        running = bool(self.backend.running)
        has_mode = self.mode_var.get() in ("signer", "hearing")
        self._toggle_armed = (not busy) and (running or has_mode)
        self._reset_armed = (not busy) and running
        self.toggle_btn.configure(text="Apagar" if running else "Encender")
        self._paint_btn(self.toggle_btn, live=self._toggle_armed, role="stop" if running else "start")
        self._paint_btn(self.reset_btn, live=self._reset_armed, role="secondary")

    def _set_dot(self, color: str) -> None:
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _start_on_main(self) -> None:
        """Arranca CUDA + API en el hilo de Tk. Un hilo extra se cuelga en Windows."""
        mode = self.mode_var.get()
        if mode not in ("signer", "hearing"):
            self._idle("Elegí un modo", error=True)
            return
        self.backend.mode = mode
        try:
            self.backend.start()
            self._idle("Cargando modelos…")
        except Exception as exc:
            self._idle(f"No arrancó: {exc}", error=True)

    def _on_toggle(self) -> None:
        if not self._toggle_armed:
            return
        if self.backend.running:
            self._busy("Apagando…")
            self.root.update_idletasks()
            try:
                self.backend.stop()
            except Exception:
                pass
            self._idle("Apagado")
            self._on_mode_change()
            return
        self._busy("Encendiendo motor…")
        self.root.update_idletasks()
        self._start_on_main()

    def _on_reset(self) -> None:
        if not self._reset_armed:
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
        self._sync_actions(busy=True)

    def _idle(self, label: str, error: bool = False) -> None:
        self.status_var.set(label)
        on = self.backend.running and not error
        self._set_dot(OK if on else BAD)
        self._sync_actions()

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
                    mode = data.get("mode") or self.backend.mode
                    if mode == "hearing":
                        bits = ["En marcha", "oyente", "voz → subtítulos"]
                        self._set_dot(OK)
                    else:
                        bits = ["En marcha", "sordo"]
                        if data.get("classifier_ready"):
                            bits.append("clasificador")
                        if data.get("semantic_ready"):
                            bits.append("LLM")
                        elif self.args.no_llm:
                            bits.append("sin LLM")
                        else:
                            bits.append("LLM cargando")
                        self._set_dot(OK if data.get("semantic_ready") or self.args.no_llm else WARN)
                    device = data.get("device") or ""
                    self.status_var.set(" · ".join(bits))
                    extra = f"{self.args.host}:{self.args.port}"
                    if device:
                        extra += f"  ·  {device}"
                    self.detail_var.set(extra)
                    self._sync_actions()
            except Exception:
                if self.backend.running:
                    self.status_var.set("Arrancando…")
                    self._set_dot(WARN)
                    self._sync_actions(busy=True)
        else:
            cur = self.status_var.get()
            if cur.startswith("En marcha") or cur == "Arrancando…":
                if self.mode_var.get():
                    self._on_mode_change()
                else:
                    self.status_var.set("Elegí un modo")
                    self._set_dot(BAD)
                    self.detail_var.set("")
                    self._sync_actions()
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
        win.title("ILSA · registros")
        win.configure(bg=BG)
        win.geometry("520x320")
        win.protocol("WM_DELETE_WINDOW", lambda: self._toggle_logs())
        text = tk.Text(
            win,
            bg=LOG_BG,
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
    """Abre la ventana ILSA y arranca el motor en segundo plano."""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    IlsaWindow(args).run()
