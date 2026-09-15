"""Formato de glosas LSA: compactar deletreo/cifras y reglas de orden/género."""
from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[¿?¡!.,;\"']")
_LETTER_RUN_RE = re.compile(r"([a-zA-ZñÑ])\1{2,}")
_SINGLE_GLOSS_LETTERS = {"X", "Ñ"}


def split_glosas(glosas: list[str] | str) -> list[str]:
    """Acepta lista o string separado por espacios y devuelve tokens no vacíos."""
    if isinstance(glosas, str):
        return [token for token in glosas.split() if token]
    return [str(token) for token in glosas if str(token).strip()]


def join_glosas(glosas: list[str] | str) -> str:
    return " ".join(split_glosas(glosas))


def unificar_glosas_dactilologicas(glosas: list[str] | str) -> list[str]:
    """Junta letras sueltas en un nombre (`M A I T E` → `Maite`) y dígitos en un número.

    Tres letras iguales seguidas se recortan a dos (`Guerra` no queda `Guerrrra`).
    Los dígitos no se recortan. `X` y `Ñ` de una sola letra se dejan en mayúscula.
    """
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


_GENDER = {"HOMBRE", "MUJER"}
_POSSESSIVES = {"MIO", "TUYO"}
_CLOSED = {
    "YO",
    "VOS",
    "EL",
    "ELLOS",
    "NOSOTROS",
    "HOMBRE",
    "MUJER",
    "MIO",
    "TUYO",
    "PASADO",
    "FUTURO",
    "AYER",
    "YA",
    "NO",
    "QUE",
    "DONDE",
    "QUIEN",
    "COMO",
    "CUANDO",
    "CUANTOS",
    "NUMERO",
    "TENER",
    "VER",
    "LLAMAR",
    "VIVIR",
    "VIVIR_EN",
    "LLEVAR",
    "ROBAR",
    "PODER",
    "REPETIR",
    "LASTIMAR",
}

_KINSHIP = (
    ("HIJO", (r"\bhijas?\b",), (r"\bhijos?\b",)),
    ("HERMANO", (r"\bhermanas?\b",), (r"\bhermanos?\b",)),
    ("ESPOSO", (r"\besposas?\b",), (r"\besposos?\b",)),
)


def _es_deletreo_o_cifra(token: str) -> bool:
    if token.isdigit():
        return True
    if len(token) == 1:
        return True
    if token[:1].isupper() and any(c.islower() for c in token[1:]):
        return True
    return False


def _sin_tildes(token: str) -> str:
    nfd = unicodedata.normalize("NFD", token)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _normalizar_token_lexico(token: str) -> str:
    if _es_deletreo_o_cifra(token):
        return token
    return _sin_tildes(token).upper()


def _tiene(spanish: str, pattern: str) -> bool:
    return re.search(pattern, spanish, flags=re.IGNORECASE) is not None


def _set_gender_after(tokens: list[str], lemma: str, gender: str | None) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok == lemma:
            j = i + 1
            while j < len(tokens) and tokens[j] in _GENDER:
                j += 1
            if gender:
                out.append(gender)
            i = j
            continue
        i += 1
    return out


def _es_sustantivo(token: str) -> bool:
    if token in _CLOSED or token in _GENDER or token in _POSSESSIVES:
        return False
    if _es_deletreo_o_cifra(token) and not token.isupper():
        return False
    return token.isupper() or (token[:1].isupper() and token[1:].islower())


def _ordenar_posesivos(tokens: list[str]) -> list[str]:
    """Pasa de MIO CUCHILLO a CUCHILLO MIO (género en el medio si hay)."""
    out = list(tokens)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out) - 1:
            if (
                out[i] in _POSSESSIVES
                and out[i + 1] in _GENDER
                and i + 2 < len(out)
                and _es_sustantivo(out[i + 2])
            ):
                out[i : i + 3] = [out[i + 2], out[i + 1], out[i]]
                changed = True
                i += 3
                continue
            if out[i] in _POSSESSIVES and _es_sustantivo(out[i + 1]):
                out[i], out[i + 1] = out[i + 1], out[i]
                changed = True
                i += 2
                continue
            i += 1
    return out


def _aplicar_parentesco(spanish: str, tokens: list[str]) -> list[str]:
    for lemma, fem_pats, masc_pats in _KINSHIP:
        if lemma not in tokens:
            continue
        if any(_tiene(spanish, p) for p in fem_pats):
            tokens = _set_gender_after(tokens, lemma, "MUJER")
        elif any(_tiene(spanish, p) for p in masc_pats):
            tokens = _set_gender_after(tokens, lemma, "HOMBRE")
    return tokens


def _aplicar_pronombre_genero(spanish: str, tokens: list[str]) -> list[str]:
    """Ellos/él explícitos llevan género; si el sujeto está tácito, no."""
    if "ELLOS" in tokens:
        if _tiene(spanish, r"\bellas\b"):
            tokens = _set_gender_after(tokens, "ELLOS", "MUJER")
        elif _tiene(spanish, r"\bellos\b"):
            tokens = _set_gender_after(tokens, "ELLOS", "HOMBRE")
        else:
            tokens = _set_gender_after(tokens, "ELLOS", None)
    if "EL" in tokens:
        if _tiene(spanish, r"\bella\b"):
            tokens = _set_gender_after(tokens, "EL", "MUJER")
        elif _tiene(spanish, r"\bél\b"):
            tokens = _set_gender_after(tokens, "EL", "HOMBRE")
        else:
            tokens = _set_gender_after(tokens, "EL", None)
    if "NOSOTROS" in tokens:
        if _tiene(spanish, r"\bnosotras\b"):
            tokens = _set_gender_after(tokens, "NOSOTROS", "MUJER")
        else:
            tokens = _set_gender_after(tokens, "NOSOTROS", None)
    return tokens


def _aplicar_calle(spanish: str, tokens: list[str]) -> list[str]:
    if not _tiene(spanish, r"\bcalle\b") or _tiene(spanish, r"\bcasa\b"):
        return tokens
    out = ["CALLE" if t == "CASA" else t for t in tokens]
    seen_calle = False
    compact: list[str] = []
    for t in out:
        if t == "CALLE":
            if seen_calle:
                continue
            seen_calle = True
        compact.append(t)
    return compact


def canonicalize_glosses(spanish: str, glosses: list[str] | str) -> list[str]:
    """Orden, género y locativos. No compacta deletreo (eso es unificar_glosas_dactilologicas)."""
    tokens = [_normalizar_token_lexico(t) for t in split_glosas(glosses)]
    tokens = _ordenar_posesivos(tokens)
    tokens = _aplicar_parentesco(spanish, tokens)
    tokens = _aplicar_pronombre_genero(spanish, tokens)
    tokens = _aplicar_calle(spanish, tokens)
    tokens = _ordenar_posesivos(tokens)
    return tokens


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
