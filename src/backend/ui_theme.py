"""Colores de ILSA."""

BG = "#1D3E53"
BG2 = "#26516D"
TEXT = "#FFFFFF"
MUTED = "#EAF4FA"
ACCENT = "#5BCBE8"
ACCENT_FG = "#1D3E53"
BTN2 = "#1D3E53"
OK = "#5BCBE8"
BAD = "#E89A94"
WARN = "#5BCBE8"
LINE = "#1A4A63"
LOG_BG = "#163040"
DISABLED_BG = "#152A38"
DISABLED_FG = "#7A9BB0"
ROW_ACTIVE = "#1A4A63"

# `state=disabled` en Windows ignora el color; apagamos el botón a mano.
BTN_DEAD = dict(
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
BTN_LIVE = {
    "start": dict(
        bg=ACCENT,
        fg=ACCENT_FG,
        activebackground="#7AD7EE",
        activeforeground=ACCENT_FG,
        relief="raised",
        bd=3,
        highlightthickness=0,
        cursor="hand2",
    ),
    "stop": dict(
        bg=TEXT,
        fg=ACCENT_FG,
        activebackground="#EAF4FA",
        activeforeground=ACCENT_FG,
        relief="raised",
        bd=3,
        highlightthickness=0,
        cursor="hand2",
    ),
    "secondary": dict(
        bg=BTN2,
        fg=TEXT,
        activebackground=ROW_ACTIVE,
        activeforeground=TEXT,
        relief="raised",
        bd=3,
        highlightbackground=ACCENT,
        highlightthickness=1,
        cursor="hand2",
    ),
}
