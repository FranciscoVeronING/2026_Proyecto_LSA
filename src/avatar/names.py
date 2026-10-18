"""Nombre de archivo en el servidor de T-pose: igualdad exacta de stem."""

from __future__ import annotations

import re

_ACCENT = str.maketrans("ÁÉÍÓÚÜ", "AEIOUU")
_KEEP = re.compile(r"[^A-Z0-9Ñ]+")


def tpose_stem(raw: str) -> str:
    """Glosa del semántico → stem de `/tposes/{stem}.pose`."""
    text = (raw or "").strip().upper()
    if not text:
        return ""
    if text.endswith("_RA"):
        text = text[:-3]
    elif text.endswith("_A"):
        text = text[:-2]
    text = text.translate(_ACCENT)
    compact = _KEEP.sub("", text)
    if compact == "Ñ":
        return "ENIE"
    compact = compact.replace("Ñ", "N")
    if compact == "PAPA":
        return "PADRE"
    return compact


def expand_spell(raw: str) -> list[str]:
    """Parte una palabra en letras. CH y Ñ cuentan una. No aplica PAPA→PADRE."""
    text = (raw or "").strip().upper().translate(_ACCENT)
    text = re.sub(r"[\s_\-]+", "", text)
    letters: list[str] = []
    index = 0
    while index < len(text):
        if text[index : index + 2] == "CH":
            letters.append("CH")
            index += 2
            continue
        char = text[index]
        index += 1
        if char.isalnum() or char == "Ñ":
            letters.append(char)
    return letters
