"""LSA gloss normalization shared by training and evaluation."""
from __future__ import annotations

import re


def unify_fingerspelled_glosses(glosses: list[str] | str) -> list[str]:
    """Join loose letters into a word and loose digits into a number.

    Letters: at most two identical characters in a row (``AAA`` -> ``Aa``).
    Digits: every repetition is kept (``3 0 1 1`` -> ``3011``).
    A lone ``X`` or ``Ñ`` stays uppercase.
    """
    if isinstance(glosses, str):
        glosses = glosses.split()

    result: list[str] = []
    letter_buffer: list[str] = []
    digit_buffer: list[str] = []

    def flush_buffers() -> None:
        nonlocal letter_buffer, digit_buffer
        if letter_buffer:
            word = "".join(letter_buffer)
            word = re.sub(r"([a-zA-ZñÑ])\1{2,}", r"\1\1", word)
            if len(word) == 1 and word.upper() in {"X", "Ñ"}:
                result.append(word.upper())
            else:
                result.append(word.capitalize())
            letter_buffer = []
        if digit_buffer:
            result.append("".join(digit_buffer))
            digit_buffer = []

    for gloss in glosses:
        if len(gloss) == 1 and (gloss.isalpha() or gloss in {"ñ", "Ñ"}):
            if digit_buffer:
                flush_buffers()
            letter_buffer.append(gloss)
        elif len(gloss) == 1 and gloss.isdigit():
            if letter_buffer:
                flush_buffers()
            digit_buffer.append(gloss)
        else:
            flush_buffers()
            result.append(gloss)

    flush_buffers()
    return result


def gloss_text(glosses: list[str] | str) -> str:
    if isinstance(glosses, list):
        return " ".join(glosses)
    return glosses.strip()


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation, for normalized accuracy."""
    text = text.lower().strip()
    text = re.sub(r"[¿?¡!.,;\"]", "", text)
    return " ".join(text.split())


def clean_output(text: str) -> str:
    """Drop ``<think>`` blocks if the model emits them."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def matches(prediction: str, references: list[str]) -> tuple[bool, bool]:
    strict = any(prediction == ref for ref in references)
    normalized = any(normalize_text(prediction) == normalize_text(ref) for ref in references)
    return strict, normalized
