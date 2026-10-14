"""Junta letras y dígitos seguidos antes de mandarlos al GGUF.

Vive acá (no en ``core``) para que el servidor semántico no dependa de ILSA.
"""

from __future__ import annotations

from typing import Sequence


def classify_gloss(gloss: str) -> str:
    """Clasifica una glosa normalizada: digit | letter | other."""
    if not gloss or len(gloss) != 1:
        return "other"
    if gloss.isdigit():
        return "digit"
    if gloss.isalpha():
        return "letter"
    return "other"


def collapse_literal_runs(glosses: Sequence[str]) -> list[str]:
    """
    ``NOMBRE MIO P E P A`` → ``["NOMBRE", "MIO", "Pepa"]``
    ``EDAD TUYO 6 7`` → ``["EDAD", "TUYO", "67"]``
    """
    out: list[str] = []
    i = 0
    n = len(glosses)
    while i < n:
        kind = classify_gloss(glosses[i])
        if kind in {"letter", "digit"}:
            j = i + 1
            while j < n and classify_gloss(glosses[j]) == kind:
                j += 1
            run = [str(g) for g in glosses[i:j]]
            if len(run) >= 2:
                chunk = "".join(run)
                if kind == "letter" and chunk.isupper():
                    chunk = chunk.capitalize()
                out.append(chunk)
            else:
                out.append(run[0])
            i = j
            continue
        out.append(str(glosses[i]))
        i += 1
    return out
