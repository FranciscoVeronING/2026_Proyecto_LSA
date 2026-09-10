"""Ventana ILSA: modo, GGUF y el switch de uvicorn. La seña se clasifica en session."""

from __future__ import annotations

import logging
import os
import queue
import sys
import tkinter as tk
import webbrowser
from collections import deque
from tkinter import font as tkfont

from backend.log_bridge import QueueLogHandler, QueueWriter
from backend.tk_popup import ChoiceRow, attach_select, destroy_popup, open_choice_popup, widget_under
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
from semantic.models import compute_label, friendly_label, list_semantic_models

REPO_URL = "https://github.com/FranciscoVeronING/2026_Proyecto_LSA"
AUTHORS = "Maite Nigro · Francisco Veron"


class IlsaWindow:
    def __init__(self, args):
        if args.gpu:
            os.environ["LSA_USE_GPU"] = "1"

        self.args = args
        self.log_q: queue.Queue[str] = queue.Queue()
        self.log_lines: deque[str] = deque(maxlen=400)
        self._hook_logs()

        self.backend = BackendHandle(args.host, args.port, enable_llm=not args.no_llm)
        self.backend.semantic_model_id = DEFAULT_MODEL_ID
        self.log_win: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None

        self.root = tk.Tk()
        self.root.title("ILSA")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._mode_menu: tk.Toplevel | None = None
        self._model_menu: tk.Toplevel | None = None
        self._build()
        self.root.after(400, self._drain_logs)
        self.root.after(800, self._poll_health)

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

        modes = tk.Frame(pad, bg=BG)
        modes.pack(fill="x", pady=(14, 0))
        self.mode_var = tk.StringVar(value="")
        self.modes_box = modes
        mode_sel = attach_select(
            modes,
            caption="Modo",
            hint="Sordo: señas LSA → español. Oyente: tu voz → subtítulos en Meet.",
            body_font=body_f,
            tiny_font=tiny_f,
            on_click=self._toggle_mode_menu,
            title="Elegí un modo",
            title_fg=MUTED,
        )
        mode_sel.box.pack(fill="x")
        self.mode_shell = mode_sel.shell
        self.mode_title = mode_sel.title
        self.mode_hint = mode_sel.hint

        self.model_var = tk.StringVar(value=DEFAULT_MODEL_ID)
        model_sel = attach_select(
            pad,
            caption="Traductor",
            hint="Cuanto más preciso, más tarda en traducir.",
            body_font=body_f,
            tiny_font=tiny_f,
            on_click=self._toggle_model_menu,
            title=friendly_label(DEFAULT_MODEL_ID),
            title_fg=TEXT,
        )
        self.model_box = model_sel.box
        self.model_shell = model_sel.shell
        self.model_title = model_sel.title
        self.model_hint = model_sel.hint
        self.root.bind_all("<Button-1>", self._on_global_click_close_menus, add="+")

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

    def _mode_choice(self, mode: str) -> tuple[str, str]:
        if mode == "signer":
            return "Sordo", "La cámara lee las señas y las traduce a español."
        if mode == "hearing":
            return "Oyente", "Transcribe lo que decís y lo pinta en tu video de Meet."
        return "Elegí un modo", "Sordo: señas LSA → español. Oyente: tu voz → subtítulos en Meet."

    def _mode_pending(self) -> bool:
        live = self.backend.mode
        chosen = self.mode_var.get()
        return bool(
            self.backend.running
            and chosen in ("signer", "hearing")
            and chosen != live
        )

    def _paint_mode_control(self) -> None:
        mode = self.mode_var.get()
        title, hint = self._mode_choice(mode)
        if self._mode_pending():
            hint = "Cambio pendiente: dale a Reiniciar modo para aplicarlo."
        self.mode_title.configure(text=title, fg=TEXT if mode else MUTED)
        self.mode_hint.configure(text=hint)
        self.mode_shell.configure(highlightbackground=ACCENT if mode else LINE)

    def _show_translator(self) -> bool:
        return self.mode_var.get() == "signer" and not self.args.no_llm

    def _semantic_choices(self) -> list[dict]:
        try:
            return list_semantic_models()
        except Exception:
            return []

    def _paint_model_control(self) -> None:
        if not hasattr(self, "model_box"):
            return
        if not self._show_translator():
            self._close_model_menu()
            self.model_box.pack_forget()
            return
        if not self.model_box.winfo_ismapped():
            self.model_box.pack(fill="x", pady=(12, 0), after=self.modes_box)
        model_id = self.model_var.get() or DEFAULT_MODEL_ID
        items = {item["id"]: item for item in self._semantic_choices()}
        item = items.get(model_id)
        title = friendly_label(model_id)
        load = compute_label(model_id)
        if item and not item.get("available", True):
            hint = "Este traductor no está instalado (falta el archivo .gguf)."
            title = f"{title} (no está)"
        elif self.backend.running and self.backend.mode == "signer":
            live = self.backend.semantic_model_id
            if live and live != model_id:
                hint = "Cargando este traductor…"
            else:
                hint = f"Carga {load}. Un modelo más preciso tarda más."
        else:
            hint = f"Carga {load}. Se usa al encender el motor."
        self.model_title.configure(text=title, fg=TEXT)
        self.model_hint.configure(text=hint)
        self.model_shell.configure(highlightbackground=ACCENT)

    def _open_menu(self, attr: str, shell, items, selected, on_pick) -> None:
        other = "_model_menu" if attr == "_mode_menu" else "_mode_menu"
        self._close_menu(other)
        if getattr(self, attr) is not None:
            self._close_menu(attr)
            return
        setattr(
            self,
            attr,
            open_choice_popup(
                root=self.root,
                shell=shell,
                items=items,
                selected_id=selected,
                on_pick=on_pick,
                body_font=self.body_f,
                tiny_font=self.tiny_f,
                on_escape=lambda: self._close_menu(attr),
            ),
        )

    def _close_menu(self, attr: str) -> None:
        destroy_popup(getattr(self, attr, None))
        setattr(self, attr, None)

    def _close_model_menu(self) -> None:
        self._close_menu("_model_menu")

    def _close_mode_menu(self) -> None:
        self._close_menu("_mode_menu")

    def _toggle_model_menu(self) -> None:
        if not self._show_translator():
            return
        choices = self._semantic_choices()
        if not choices:
            return
        items = [
            ChoiceRow(
                item["id"],
                friendly_label(item["id"]),
                "No está descargado" if not item.get("available", True) else f"Carga {compute_label(item['id'])}",
                bool(item.get("available", True)),
            )
            for item in choices
        ]
        self._open_menu("_model_menu", self.model_shell, items, self.model_var.get(), self._pick_model)

    def _toggle_mode_menu(self) -> None:
        items = [
            ChoiceRow("signer", "Sordo", "Señas LSA → español"),
            ChoiceRow("hearing", "Oyente", "Voz → subtítulos en la cámara"),
        ]
        self._open_menu("_mode_menu", self.mode_shell, items, self.mode_var.get(), self._pick_mode)

    def _on_global_click_close_menus(self, event: tk.Event) -> None:
        pairs = (
            ("_mode_menu", self.mode_shell),
            ("_model_menu", self.model_shell),
        )
        for attr, shell in pairs:
            menu = getattr(self, attr)
            if menu is None:
                continue
            if not widget_under(event.widget, shell) and not widget_under(event.widget, menu):
                self._close_menu(attr)

    def _pick_model(self, model_id: str) -> None:
        self._close_model_menu()
        self.model_var.set(model_id)
        self.backend.semantic_model_id = model_id
        if (
            self.backend.running
            and self.backend.mode == "signer"
            and self.mode_var.get() == "signer"
        ):
            try:
                from backend import server as srv

                switch = getattr(srv._session, "switch_semantic_model", None)
                if switch:
                    switch(model_id)
                    self.status_var.set("Cargando traductor…")
                    self._set_dot(WARN)
            except Exception as exc:
                print(f"[ilsa] No se pudo cambiar el traductor: {exc}")
        self._sync_actions()

    def _pick_mode(self, mode: str) -> None:
        self._close_mode_menu()
        self.mode_var.set(mode)
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
        self._paint_model_control()

    def _set_dot(self, color: str) -> None:
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _bind_opts(self) -> str | None:
        mode = self.mode_var.get()
        if mode not in ("signer", "hearing"):
            self._idle("Elegí un modo", error=True)
            return None
        self.backend.mode = mode
        self.backend.semantic_model_id = self.model_var.get() or DEFAULT_MODEL_ID
        return mode

    def _start_on_main(self) -> None:
        # PyTorch/CUDA en el hilo de Tk; otro hilo se cuelga en Windows.
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
            return
        self._busy("Encendiendo motor…")
        self.root.update_idletasks()
        self._start_on_main()

    def _on_reset(self) -> None:
        if not self._reset_armed or not self._bind_opts():
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
                    if self._mode_pending():
                        bits.append("Reiniciá el modo")
                    elif mode == "signer" and (
                        data.get("semantic_busy")
                        or (
                            data.get("semantic_model")
                            and self.model_var.get()
                            and data.get("semantic_model") != self.model_var.get()
                        )
                    ):
                        bits.append("traductor…")
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
        self._close_mode_menu()
        self._close_model_menu()
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
    IlsaWindow(args).run()
