"""Normalización de glosas LSA compartida por el entrenamiento y la evaluación."""
from __future__ import annotations

import re


def unificar_glosas_dactilologicas(glosas: list[str] | str) -> list[str]:
    """Junta letras sueltas en una palabra y dígitos sueltos en un número.

    Letras: como máximo dos iguales seguidas (``AAA`` → ``Aa``).
    Dígitos: se conservan todas las repeticiones (``3 0 1 1`` → ``3011``).
    Una sola ``X`` o ``Ñ`` queda en mayúscula.
    """
    if isinstance(glosas, str):
        glosas = glosas.split()

    resultado: list[str] = []
    buffer_letras: list[str] = []
    buffer_digitos: list[str] = []

    def vaciar_buffers() -> None:
        nonlocal buffer_letras, buffer_digitos
        if buffer_letras:
            palabra = "".join(buffer_letras)
            palabra = re.sub(r"([a-zA-ZñÑ])\1{2,}", r"\1\1", palabra)
            if len(palabra) == 1 and palabra.upper() in {"X", "Ñ"}:
                resultado.append(palabra.upper())
            else:
                resultado.append(palabra.capitalize())
            buffer_letras = []
        if buffer_digitos:
            resultado.append("".join(buffer_digitos))
            buffer_digitos = []

    for glosa in glosas:
        if len(glosa) == 1 and (glosa.isalpha() or glosa in {"ñ", "Ñ"}):
            if buffer_digitos:
                vaciar_buffers()
            buffer_letras.append(glosa)
        elif len(glosa) == 1 and glosa.isdigit():
            if buffer_letras:
                vaciar_buffers()
            buffer_digitos.append(glosa)
        else:
            vaciar_buffers()
            resultado.append(glosa)

    vaciar_buffers()
    return resultado


def texto_glosas(glosas: list[str] | str) -> str:
    if isinstance(glosas, list):
        return " ".join(glosas)
    return glosas.strip()


def normalizar_texto(texto: str) -> str:
    """Minúsculas y sin puntuación, para el accuracy normalizado."""
    texto = texto.lower().strip()
    texto = re.sub(r"[¿?¡!.,;\"]", "", texto)
    return " ".join(texto.split())


def limpiar_salida(texto: str) -> str:
    """Quita bloques ``<think>`` si el modelo los emite."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


def coincide(prediccion: str, referencias: list[str]) -> tuple[bool, bool]:
    estricta = any(prediccion == ref for ref in referencias)
    normalizada = any(normalizar_texto(prediccion) == normalizar_texto(ref) for ref in referencias)
    return estricta, normalizada
