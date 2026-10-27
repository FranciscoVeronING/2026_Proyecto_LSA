"""Join consecutive single letters or digits before ``POST /sign``."""

from __future__ import annotations

from typing import Sequence


def _kind(token: str) -> str:
    if not token or len(token) != 1:
        return "other"
    if token.isdigit():
        return "digit"
    if token.isalpha():
        return "letter"
    return "other"


def collapse_letter_digit_runs(glosses: Sequence[str]) -> list[str]:
    """``NOMBRE MIO P E P A`` → ``NOMBRE MIO Pepa``; ``6 7`` → ``67``."""
    out: list[str] = []
    i = 0
    n = len(glosses)
    while i < n:
        kind = _kind(glosses[i])
        if kind in {"letter", "digit"}:
            j = i + 1
            while j < n and _kind(glosses[j]) == kind:
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
