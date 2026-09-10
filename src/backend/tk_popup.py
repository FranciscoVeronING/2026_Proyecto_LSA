"""Dropdown custom de ILSA (Tk no dibuja bien un Combobox oscuro)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk

from backend.ui_theme import ACCENT, BG, BG2, LINE, MUTED, ROW_ACTIVE, TEXT


@dataclass
class SelectField:
    box: tk.Frame
    shell: tk.Frame
    title: tk.Label
    hint: tk.Label


@dataclass(frozen=True)
class ChoiceRow:
    id: str
    title: str
    subtitle: str
    enabled: bool = True


def widget_under(widget, ancestor: tk.Misc) -> bool:
    w = widget
    while w:
        if w == ancestor:
            return True
        w = getattr(w, "master", None)
    return False


def destroy_popup(win: tk.Toplevel | None) -> None:
    if win is None:
        return
    try:
        win.destroy()
    except Exception:
        pass


def attach_select(parent, *, caption, hint, body_font, tiny_font, on_click, title, title_fg) -> SelectField:
    box = tk.Frame(parent, bg=BG)
    tk.Label(box, text=caption, font=tiny_font, fg=MUTED, bg=BG).pack(anchor="w")
    shell = tk.Frame(box, bg=BG2, highlightbackground=LINE, highlightthickness=1)
    shell.pack(fill="x", pady=(4, 0))
    inner = tk.Frame(shell, bg=BG2, padx=12, pady=8)
    inner.pack(fill="x")
    title_lbl = tk.Label(inner, text=title, font=body_font, fg=title_fg, bg=BG2, anchor="w")
    title_lbl.pack(side="left", fill="x", expand=True)
    chevron = tk.Label(inner, text="▾", font=body_font, fg=ACCENT, bg=BG2)
    chevron.pack(side="right")
    for w in (shell, inner, title_lbl, chevron):
        w.bind("<Button-1>", lambda _e: on_click())
        w.configure(cursor="hand2")
    hint_lbl = tk.Label(
        box, text=hint, font=tiny_font, fg=MUTED, bg=BG, justify="left", wraplength=340
    )
    hint_lbl.pack(anchor="w", pady=(6, 0))
    return SelectField(box=box, shell=shell, title=title_lbl, hint=hint_lbl)


def open_choice_popup(
    *,
    root: tk.Tk,
    shell: tk.Misc,
    items: list[ChoiceRow],
    selected_id: str,
    on_pick: Callable[[str], None],
    body_font,
    tiny_font,
    on_escape: Callable[[], None],
) -> tk.Toplevel:
    menu = tk.Toplevel(root)
    menu.overrideredirect(True)
    menu.configure(bg=LINE)
    root.update_idletasks()
    x = shell.winfo_rootx()
    y = shell.winfo_rooty() + shell.winfo_height() + 4
    w = max(shell.winfo_width(), 240)
    menu.geometry(f"{w}x1+{x}+{y}")
    body = tk.Frame(menu, bg=BG2)
    body.pack(fill="both", expand=True, padx=1, pady=1)

    def hover(row, title, sub, active):
        bg = ROW_ACTIVE if active else BG2
        row.configure(bg=bg)
        title.configure(bg=bg)
        sub.configure(bg=bg)

    for item in items:
        selected = selected_id == item.id
        row_bg = ROW_ACTIVE if selected else BG2
        row = tk.Frame(body, bg=row_bg, padx=12, pady=8)
        row.pack(fill="x")
        title = tk.Label(
            row,
            text=item.title,
            font=body_font,
            fg=ACCENT if selected else (MUTED if not item.enabled else TEXT),
            bg=row_bg,
            anchor="w",
        )
        title.pack(fill="x")
        sub = tk.Label(row, text=item.subtitle, font=tiny_font, fg=MUTED, bg=row_bg, anchor="w")
        sub.pack(fill="x")
        for w in (row, title, sub):
            w.configure(cursor="hand2" if item.enabled else "arrow")
            if not item.enabled:
                continue
            w.bind("<Button-1>", lambda _e, v=item.id: on_pick(v))
            w.bind("<Enter>", lambda _e, r=row, a=title, b=sub: hover(r, a, b, True))
            w.bind("<Leave>", lambda _e, r=row, a=title, b=sub, sel=selected: hover(r, a, b, sel))

    menu.update_idletasks()
    menu.geometry(f"{w}x{body.winfo_reqheight() + 2}+{x}+{y}")
    menu.bind("<Escape>", lambda _e: on_escape())
    menu.focus_set()
    return menu
