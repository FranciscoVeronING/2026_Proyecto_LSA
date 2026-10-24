"""Ventana ILSA: modo y el switch de uvicorn. La seña se clasifica en session."""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
import webbrowser
from collections import deque
from pathlib import Path
from tkinter import font as tkfont

from backend.log_bridge import QueueLogHandler, QueueWriter
from backend.ui_theme import (
    ACCENT,
    BAD,
    BG,
    BG2,
    BTN_DEAD,
    BTN_LIVE,
    LINE,
    LOG_BG,
    MUTED,
    OK,
    TEXT,
    WARN,
)
from backend.uvicorn_handle import BackendHandle
from semantic.config import DEFAULT_MODEL_ID

REPO_URL = "https://github.com/FranciscoVeronING/2026_Proyecto_LSA"
AUTHORS = "Maite Nigro · Francisco Veron"
WIN_W = 420
WIN_H = 620
SENARIO_1_URL = (
    "https://cas.org.ar/senario-de-terminos-y-expresiones-en-lengua-de-senas-argentina/"
)
SENARIO_2_URL = (
    "https://cas.org.ar/senario-de-terminos-y-expresiones-en-lengua-de-senas-argentina-vol-2/"
)
LSA_FAMILIA_URL = "https://play.google.com/store/apps/details?id=com.proyectossolidarios&hl=es-AR"


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


class IlsaWindow:
    def __init__(self, args):
        self.args = args
        self.log_q: queue.Queue[str] = queue.Queue()
        self.log_lines: deque[str] = deque(maxlen=400)
        self._hook_logs()

        self.backend = BackendHandle(args.host, args.port, enable_llm=not args.no_llm)
        self.backend.semantic_model_id = DEFAULT_MODEL_ID
        self.log_win: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None
        self._boot_q: queue.Queue = queue.Queue()
        self._ready = False
        self._pad = None
        self._after_cloud: str | None = None
        self._booting = False
        self._icon_images: list[tk.PhotoImage] = []

        self.root = tk.Tk()
        self.root.title("ILSA")
        self.root.configure(bg=BG)
        self._set_window_icon()
        self.root.resizable(False, False)
        self.root.minsize(WIN_W, WIN_H)
        self.root.maxsize(WIN_W, WIN_H)
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        body_f = tkfont.Font(family="Segoe UI", size=10)
        self._splash = tk.Frame(self.root, bg=BG)
        self._splash.pack(fill="both", expand=True)
        splash_inner = tk.Frame(self._splash, bg=BG)
        splash_inner.place(relx=0.5, rely=0.5, anchor="center")
        splash_logo = self._load_photo(5)
        if splash_logo is not None:
            tk.Label(splash_inner, image=splash_logo, bg=BG, bd=0).pack(pady=(0, 18))
        self._spin_canvas = tk.Canvas(
            splash_inner, width=44, height=44, bg=BG, highlightthickness=0
        )
        self._spin_canvas.pack()
        self._spin_angle = 0
        self._spin_job = None
        self._splash_status = tk.StringVar(value="Conectando con el servidor…")
        tk.Label(
            splash_inner,
            textvariable=self._splash_status,
            font=body_f,
            fg=TEXT,
            bg=BG,
            justify="center",
        ).pack(pady=(16, 0))
        self._retry_btn = tk.Button(
            splash_inner,
            text="Reintentar",
            font=body_f,
            padx=16,
            pady=8,
            command=self._start_bootstrap,
            **BTN_LIVE["start"],
        )
        self._tick_spin()
        self.root.after(200, self._drain_logs)
        self.root.after(80, self._cloud_check)

    def _logo_png(self) -> Path | None:
        root = _app_root()
        for path in (
            root / "assets" / "ilsa-logo.png",
            root / "extension" / "icons" / "icon128.png",
        ):
            if path.is_file():
                return path
        return None

    def _load_photo(self, subsample: int) -> tk.PhotoImage | None:
        path = self._logo_png()
        if path is None:
            return None
        try:
            img = tk.PhotoImage(file=str(path))
            if subsample > 1:
                img = img.subsample(subsample, subsample)
            self._icon_images.append(img)
            return img
        except tk.TclError:
            return None

    def _set_window_icon(self) -> None:
        root = _app_root()
        ico = root / "ilsa.ico"
        if not ico.is_file():
            ico = root / "packaging" / "ilsa.ico"
        try:
            if ico.is_file():
                self.root.iconbitmap(default=str(ico))
            elif getattr(sys, "frozen", False):
                self.root.iconbitmap(default=sys.executable)
        except tk.TclError:
            pass
        photo = self._load_photo(8)
        if photo is not None:
            try:
                self.root.iconphoto(True, photo)
            except tk.TclError:
                pass

    def _tick_spin(self) -> None:
        if self._splash is None:
            return
        try:
            if not self._splash.winfo_ismapped():
                self._spin_job = None
                return
        except tk.TclError:
            self._spin_job = None
            return
        try:
            if not self._spin_canvas.winfo_exists():
                return
        except tk.TclError:
            return
        c = self._spin_canvas
        c.delete("all")
        c.create_arc(
            4,
            4,
            40,
            40,
            start=self._spin_angle,
            extent=300,
            style="arc",
            outline=ACCENT,
            width=4,
        )
        self._spin_angle = (self._spin_angle + 14) % 360
        self._spin_job = self.root.after(40, self._tick_spin)

    def _stop_spin(self) -> None:
        if self._spin_job is not None:
            try:
                self.root.after_cancel(self._spin_job)
            except Exception:
                pass
            self._spin_job = None

    def _cloud_check(self, after: str | None = None) -> None:
        if after is not None:
            self._after_cloud = after
        self._show_splash()
        self._start_bootstrap()

    def _splash_visible(self) -> bool:
        if self._splash is None:
            return False
        try:
            return bool(self._splash.winfo_ismapped())
        except tk.TclError:
            return False

    def _show_splash(self, msg: str = "Comprobando servidores…") -> None:
        self._splash_status.set(msg)
        try:
            self._retry_btn.pack_forget()
        except tk.TclError:
            pass
        if self._pad is not None:
            self._pad.pack_forget()
        if self._splash is not None and not self._splash.winfo_ismapped():
            self._splash.pack(fill="both", expand=True)
        self._stop_spin()
        self._tick_spin()

    def _start_bootstrap(self) -> None:
        if self._booting:
            return
        self._booting = True
        self._retry_btn.pack_forget()
        self._splash_status.set("Comprobando traductor (puede tardar unos minutos)…")
        if self._spin_job is None:
            self._tick_spin()
        threading.Thread(target=self._bootstrap_worker, daemon=True, name="ilsa-semantic").start()
        self.root.after(150, self._poll_boot)

    def _bootstrap_worker(self) -> None:
        try:
            from avatar.remote import ping as ping_avatar
            from semantic.remote import configured, ping as ping_semantic

            if not self.args.no_llm:
                if not configured():
                    raise RuntimeError("Falta la URL del traductor semántico")
                self._boot_q.put(("status", "Comprobando traductor (puede tardar unos minutos)…"))
                if not ping_semantic():
                    raise RuntimeError("El traductor semántico no responde")
            self._boot_q.put(("status", "Comprobando avatar…"))
            if not ping_avatar(timeout=20.0):
                raise RuntimeError("El servidor de T-pose no responde")
            self._boot_q.put(("ok",))
        except Exception as exc:
            self._boot_q.put(("err", str(exc)))

    def _poll_boot(self) -> None:
        try:
            while True:
                item = self._boot_q.get_nowait()
                kind = item[0]
                if kind == "status":
                    self._splash_status.set(item[1])
                elif kind == "ok":
                    self._booting = False
                    self._finish_cloud_ok()
                    return
                elif kind == "err":
                    self._booting = False
                    self._stop_spin()
                    self._splash_status.set(item[1] if len(item) > 1 else "No se pudo conectar")
                    self._retry_btn.pack(pady=(16, 0))
                    return
        except queue.Empty:
            pass
        if self._booting:
            self.root.after(150, self._poll_boot)

    def _finish_cloud_ok(self) -> None:
        self._show_main()
        after = self._after_cloud
        self._after_cloud = None
        if after == "start":
            self._start_on_main()
        elif after == "restart":
            self._restart_on_main()

    def _show_main(self) -> None:
        self._stop_spin()
        if self._splash is not None:
            self._splash.pack_forget()
        if self._pad is None:
            self._build()
            self.root.geometry(f"{WIN_W}x{WIN_H}")
            self.root.after(800, self._poll_health)
        else:
            self._pad.pack(fill="both", expand=True)
        self._ready = True

    def _hook_logs(self) -> None:
        writer = QueueWriter(self.log_q)
        sys.stdout = writer
        sys.stderr = writer
        handler = QueueLogHandler(self.log_q)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger("uvicorn").addHandler(handler)
        logging.getLogger("uvicorn.error").addHandler(handler)

    def _build(self) -> None:
        pad = tk.Frame(self.root, bg=BG, padx=28, pady=22)
        self._pad = pad
        pad.pack(fill="both", expand=True)

        title_f = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        body_f = tkfont.Font(family="Segoe UI", size=10)
        small_f = tkfont.Font(family="Segoe UI", size=9)
        tiny_f = tkfont.Font(family="Segoe UI", size=8)

        head = tk.Frame(pad, bg=BG)
        head.pack(fill="x")
        title_logo = self._load_photo(10)
        if title_logo is not None:
            tk.Label(head, image=title_logo, bg=BG, bd=0).pack(side="left", padx=(0, 10))
        tk.Label(head, text="ILSA", font=title_f, fg=TEXT, bg=BG).pack(side="left")
        info_btn = tk.Button(
            head,
            text="i",
            font=body_f,
            width=3,
            command=self._open_sources,
            **BTN_LIVE["secondary"],
        )
        info_btn.pack(side="right")
        tk.Label(
            pad,
            text="Intérprete de Lengua de Señas Argentina",
            font=small_f,
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(0, 12))

        self.body_f = body_f
        self.tiny_f = tiny_f

        tk.Label(
            pad,
            text="Elegí el modo y después encendé el motor.",
            font=body_f,
            fg=TEXT,
            bg=BG,
            justify="left",
        ).pack(anchor="w")

        self.mode_var = tk.StringVar(value="")
        self.mode_hint_var = tk.StringVar(
            value="Sordo: señas LSA → español. Oyente: tu voz → subtítulos en Meet."
        )
        modes = tk.Frame(pad, bg=BG)
        modes.pack(fill="x", pady=(14, 0))
        tk.Label(modes, text="Modo", font=tiny_f, fg=MUTED, bg=BG).pack(anchor="w")
        row_modes = tk.Frame(modes, bg=BG)
        row_modes.pack(fill="x", pady=(6, 0))
        self.signer_btn = tk.Button(
            row_modes,
            text="Sordo",
            font=body_f,
            padx=16,
            pady=10,
            command=lambda: self._pick_mode("signer"),
        )
        self.signer_btn.pack(side="left", fill="x", expand=True)
        self.hearing_btn = tk.Button(
            row_modes,
            text="Oyente",
            font=body_f,
            padx=16,
            pady=10,
            command=lambda: self._pick_mode("hearing"),
        )
        self.hearing_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        tk.Label(
            modes,
            textvariable=self.mode_hint_var,
            font=tiny_f,
            fg=MUTED,
            bg=BG,
            justify="left",
            wraplength=360,
            width=48,
            height=3,
            anchor="nw",
        ).pack(anchor="w", pady=(8, 0))

        self.handed_var = tk.StringVar(value="right")
        self.handed_frame = tk.Frame(modes, bg=BG)
        tk.Label(self.handed_frame, text="Mano", font=tiny_f, fg=MUTED, bg=BG).pack(anchor="w")
        row_hand = tk.Frame(self.handed_frame, bg=BG)
        row_hand.pack(fill="x", pady=(6, 0))
        self.zurdo_btn = tk.Button(
            row_hand,
            text="Zurdo",
            font=body_f,
            padx=16,
            pady=10,
            command=lambda: self._pick_handed("left"),
        )
        self.zurdo_btn.pack(side="left", fill="x", expand=True)
        self.diestro_btn = tk.Button(
            row_hand,
            text="Diestro",
            font=body_f,
            padx=16,
            pady=10,
            command=lambda: self._pick_handed("right"),
        )
        self.diestro_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

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
            width=12,
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
            width=16,
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
        tk.Label(
            pad,
            textvariable=self.detail_var,
            font=tiny_f,
            fg=MUTED,
            bg=BG,
            wraplength=360,
            width=48,
            height=2,
            justify="left",
            anchor="nw",
        ).pack(anchor="w", pady=(6, 0))

    def _link_label(self, parent, text: str, url: str, font) -> None:
        lab = tk.Label(parent, text=text, font=font, fg=ACCENT, bg=BG, cursor="hand2", justify="left", wraplength=340)
        lab.pack(anchor="w", pady=(4, 0))
        lab.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))

    def _open_sources(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Fuentes de las señas")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.transient(self.root)
        body = tk.Frame(win, bg=BG, padx=24, pady=20)
        body.pack(fill="both")
        title_f = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        body_f = tkfont.Font(family="Segoe UI", size=10)
        small_f = tkfont.Font(family="Segoe UI", size=9)
        tk.Label(body, text="Fuentes de las señas", font=title_f, fg=TEXT, bg=BG).pack(anchor="w")
        tk.Label(
            body,
            text=(
                "Las señas que toma ILSA siguen la implementación del Señario CAS "
                "(tomos 1 y 2). Si alguna seña no está ahí, recurrimos a la aplicación "
                "LSA en familia."
            ),
            font=body_f,
            fg=TEXT,
            bg=BG,
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(10, 8))
        self._link_label(body, "Señario CAS · tomo 1", SENARIO_1_URL, small_f)
        self._link_label(body, "Señario CAS · tomo 2", SENARIO_2_URL, small_f)
        self._link_label(body, "LSA en familia (Play Store)", LSA_FAMILIA_URL, small_f)
        close = tk.Button(body, text="Cerrar", font=body_f, padx=16, pady=6, command=win.destroy, **BTN_LIVE["secondary"])
        close.pack(anchor="e", pady=(18, 0))
        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - h) // 2
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.grab_set()
        win.focus_set()

    def _mode_choice(self, mode: str) -> tuple[str, str]:
        if mode == "signer":
            return "Sordo", "La cámara lee las señas y las traduce a español."
        if mode == "hearing":
            return "Oyente", "Transcribe lo que decís y lo pinta en tu video de Meet."
        return "Elegí un modo", "Sordo: señas LSA → español. Oyente: tu voz → subtítulos en Meet."

    def _mode_pending(self) -> bool:
        live = self.backend.mode
        chosen = self.mode_var.get()
        mode_changed = bool(
            self.backend.running
            and chosen in ("signer", "hearing")
            and chosen != live
        )
        handed_changed = bool(
            self.backend.running
            and chosen == "signer"
            and live == "signer"
            and self._want_left_handed() != bool(self.backend.left_handed)
        )
        return mode_changed or handed_changed

    def _want_left_handed(self) -> bool:
        return self.handed_var.get() == "left"

    def _paint_handed_control(self) -> None:
        if self.mode_var.get() != "signer":
            self.handed_frame.pack_forget()
            return
        if not self.handed_frame.winfo_ismapped():
            self.handed_frame.pack(fill="x", pady=(10, 0))
        if self.handed_var.get() not in ("left", "right"):
            self.handed_var.set("right")
        handed = self.handed_var.get()
        self._paint_btn(self.zurdo_btn, live=True, role="start" if handed == "left" else "secondary")
        self._paint_btn(self.diestro_btn, live=True, role="start" if handed == "right" else "secondary")

    def _paint_mode_control(self) -> None:
        mode = self.mode_var.get()
        _, hint = self._mode_choice(mode)
        if self._mode_pending():
            hint = "Cambio pendiente: dale a Reiniciar modo para aplicarlo."
        self.mode_hint_var.set(hint)
        self._paint_btn(self.signer_btn, live=True, role="start" if mode == "signer" else "secondary")
        self._paint_btn(self.hearing_btn, live=True, role="start" if mode == "hearing" else "secondary")
        self._paint_handed_control()

    def _pick_handed(self, handed: str) -> None:
        if handed not in ("left", "right"):
            handed = "right"
        self.handed_var.set(handed)
        self._on_mode_change()

    def _pick_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        if mode == "signer" and self.handed_var.get() not in ("left", "right"):
            self.handed_var.set("right")
        self._on_mode_change()

    def _on_mode_change(self) -> None:
        mode = self.mode_var.get()
        if self._mode_pending():
            self.status_var.set("Modo cambiado · Reiniciá para aplicar")
        elif not self.backend.running:
            if mode == "signer":
                self.status_var.set("Listo · modo sordo")
            elif mode == "hearing":
                self.status_var.set("Listo · modo oyente")
        self._sync_actions()

    def _paint_btn(self, btn: tk.Button, *, live: bool, role: str) -> None:
        btn.configure(**(BTN_DEAD if not live else BTN_LIVE[role]))

    def _sync_actions(self, busy: bool = False) -> None:
        running = bool(self.backend.running)
        has_mode = self.mode_var.get() in ("signer", "hearing")
        self._toggle_armed = (not busy) and (running or has_mode)
        self._reset_armed = (not busy) and running
        pending = self._mode_pending()
        self.toggle_btn.configure(text="Apagar" if running else "Encender")
        self.reset_btn.configure(text="Reiniciar modo" if pending else "Reiniciar")
        self._paint_btn(self.toggle_btn, live=self._toggle_armed, role="stop" if running else "start")
        self._paint_btn(
            self.reset_btn,
            live=self._reset_armed,
            role="start" if pending else "secondary",
        )
        self._paint_mode_control()

    def _set_dot(self, color: str) -> None:
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _bind_opts(self) -> str | None:
        mode = self.mode_var.get()
        if mode not in ("signer", "hearing"):
            self._idle("Elegí un modo", error=True)
            return None
        self.backend.mode = mode
        self.backend.semantic_model_id = DEFAULT_MODEL_ID
        self.backend.left_handed = mode == "signer" and self._want_left_handed()
        return mode

    def _start_on_main(self) -> None:
        if not self._bind_opts():
            return
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
            self._cloud_check()
            return
        self._cloud_check("start")

    def _restart_on_main(self) -> None:
        if not self._bind_opts():
            return
        self._busy("Reiniciando…")
        self.root.update_idletasks()
        try:
            self.backend.restart()
            self._idle("Cargando modelos…")
        except Exception as exc:
            self._idle(f"No reinició: {exc}", error=True)

    def _on_reset(self) -> None:
        if not self._reset_armed or not self._bind_opts():
            return
        self._cloud_check("restart")

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
        if self._splash_visible():
            self.root.after(1000, self._poll_health)
            return
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
                    semantic_ok = bool(data.get("semantic_ready")) or self.args.no_llm
                    if not semantic_ok:
                        self.status_var.set("Traductor caído · Reiniciá ILSA")
                        self._set_dot(BAD)
                        self.detail_var.set(data.get("semantic_error") or "El semántico no responde")
                        self._sync_actions()
                    else:
                        if mode == "hearing":
                            bits = ["En marcha", "oyente", "voz → subtítulos"]
                            self._set_dot(OK)
                        else:
                            bits = ["En marcha", "sordo"]
                            if data.get("classifier_ready"):
                                bits.append("clasificador")
                            bits.append("traductor")
                            self._set_dot(OK)
                        if self._mode_pending():
                            bits.append("Reiniciá el modo")
                        elif mode == "signer" and data.get("semantic_busy"):
                            bits.append("traduciendo…")
                        self.status_var.set(" · ".join(bits))
                        extra = f"{self.args.host}:{self.args.port}  ·  CPU"
                        line = data.get("metrics") or ""
                        if line:
                            extra = f"{extra}  ·  {line}"
                        self.detail_var.set(extra)
                        self._sync_actions()
            except Exception:
                if self.backend.running:
                    self._set_dot(WARN)
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
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    IlsaWindow(args).run()
