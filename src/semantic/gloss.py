"""Join gloss tokens, clean model output, and normalize text for scoring."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[¿?¡!.,;\"']")


def split_glosses(glosses: list[str] | str) -> list[str]:
    if isinstance(glosses, str):
        return [token for token in glosses.split() if token]
    return [str(token) for token in glosses if str(token).strip()]


def join_glosses(glosses: list[str] | str) -> str:
    return " ".join(split_glosses(glosses))


def normalize_for_metrics(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text
