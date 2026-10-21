"""Helpers para unir tokens de glosas y limpiar texto de salida."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[¿?¡!.,;\"']")


def split_glosas(glosas: list[str] | str) -> list[str]:
    if isinstance(glosas, str):
        return [token for token in glosas.split() if token]
    return [str(token) for token in glosas if str(token).strip()]


def join_glosas(glosas: list[str] | str) -> str:
    return " ".join(split_glosas(glosas))


def normalizar_para_metricas(texto: str) -> str:
    texto = texto.lower().strip()
    texto = _PUNCT_RE.sub("", texto)
    return " ".join(texto.split())


def limpiar_salida_modelo(texto: str) -> str:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    texto = texto.strip()
    if "\n" in texto:
        texto = texto.split("\n", 1)[0].strip()
    return texto
