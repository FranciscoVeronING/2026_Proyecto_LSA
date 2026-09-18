"""Compacta deletreo y cifras al cargar el dataset o al puntuar una predicción."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[¿?¡!.,;\"']")
_LETTER_RUN_RE = re.compile(r"([a-zA-ZñÑ])\1{2,}")
_SINGLE_GLOSS_LETTERS = {"X", "Ñ"}


def split_glosas(glosas: list[str] | str) -> list[str]:
    if isinstance(glosas, str):
        return [token for token in glosas.split() if token]
    return [str(token) for token in glosas if str(token).strip()]


def join_glosas(glosas: list[str] | str) -> str:
    return " ".join(split_glosas(glosas))


def unificar_glosas_dactilologicas(glosas: list[str] | str) -> list[str]:
    """`M A I T E` → `Maite`, dígitos sueltos → un número. `X` y `Ñ` se dejan."""
    tokens = split_glosas(glosas)
    resultado: list[str] = []
    buffer_letras: list[str] = []
    buffer_digitos: list[str] = []

    def vaciar_buffers() -> None:
        nonlocal buffer_letras, buffer_digitos
        if buffer_letras:
            palabra = "".join(buffer_letras)
            palabra = _LETTER_RUN_RE.sub(r"\1\1", palabra)
            if len(palabra) == 1 and palabra.upper() in _SINGLE_GLOSS_LETTERS:
                resultado.append(palabra.upper())
            else:
                resultado.append(palabra.capitalize())
            buffer_letras = []
        if buffer_digitos:
            resultado.append("".join(buffer_digitos))
            buffer_digitos = []

    for token in tokens:
        if len(token) == 1 and (token.isalpha() or token in {"ñ", "Ñ"}):
            if buffer_digitos:
                vaciar_buffers()
            buffer_letras.append(token)
        elif len(token) == 1 and token.isdigit():
            if buffer_letras:
                vaciar_buffers()
            buffer_digitos.append(token)
        else:
            vaciar_buffers()
            resultado.append(token)

    vaciar_buffers()
    return resultado


def formato_canonico(glosas: list[str] | str) -> str:
    return join_glosas(unificar_glosas_dactilologicas(glosas))


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
