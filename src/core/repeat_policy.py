"""Repeat-gloss acceptance and literal-sequence formatting."""

from __future__ import annotations

from typing import Optional, Sequence


# Letters the classifier swaps with each other or with a lexical sign:
# I, T and OJO share almost the same handshape and differ by where they rest
# (cheek / below the mouth / next to the eye).
# A fingerspell that contains them needs the translator's judgment.
AMBIGUOUS_LETTERS = frozenset({"I", "T"})

# Lexical signs the classifier confuses with a letter (same handshape).
# Next to or between letters, they almost certainly meant to spell the letter.
LETTER_LOOKALIKES = {
    "OJO": "I",
    "VIERNES": "V",
    "AÑOS": "G",
}


def classify_gloss(gloss: str) -> str:
    """Classify a normalized gloss: digit | letter | other."""
    if not gloss or len(gloss) != 1:
        return "other"
    if gloss.isdigit():
        return "digit"
    if gloss.isalpha():
        return "letter"
    return "other"


def resolve_letter_lookalikes(glosses: Sequence[str]) -> list[str]:
    """
    OJO / VIERNES / AÑOS next to letters → I / V / G.

    Several passes: ``M OJO VIERNES A`` first rewrites VIERNES (next to A)
    then OJO (next to V). A lone OJO or AÑOS stays as a sign.
    """
    out = [str(g) for g in glosses]
    changed = True
    while changed:
        changed = False
        nxt = list(out)
        for i, tok in enumerate(out):
            letter = LETTER_LOOKALIKES.get(tok.upper())
            if not letter:
                continue
            left = out[i - 1] if i > 0 else ""
            right = out[i + 1] if i + 1 < len(out) else ""
            if classify_gloss(left) != "letter" and classify_gloss(right) != "letter":
                continue
            nxt[i] = letter
            changed = True
        out = nxt
    return out


def collapse_literal_runs(glosses: Sequence[str]) -> list[str]:
    """
    Join **consecutive** letters or digits before sending them to the translator.

    ``NOMBRE MIO P E P A`` → ``["NOMBRE", "MIO", "Pepa"]``
    ``EDAD TUYO 6 7`` → ``["EDAD", "TUYO", "67"]``

    A lone letter or digit is left as-is (it may be a gloss, not fingerspelling).
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


def format_literal_utterance(
    glosses: Sequence[str],
    ambiguous_letters: frozenset = AMBIGUOUS_LETTERS,
) -> Optional[str]:
    """
    If the utterance is only fingerspelling or only digits, build the display text.
    Avoid depending on the translator (which sometimes returns empty).

    Returns None when the translator should decide with context.

    Note on O/0 and 2/V: no extra exception here. If the sequence is all one
    kind, the character is already the right kind; if the classifier picked the
    wrong kind, the sequence is mixed and still falls through to the translator.
    """
    if not glosses:
        return None

    kinds = [classify_gloss(g) for g in glosses]

    if all(k == "digit" for k in kinds):
        return f"{''.join(glosses)}."

    if all(k == "letter" for k in kinds):
        # A single letter may actually be a misclassified lexical sign.
        if len(glosses) == 1:
            return None
        # I and T are interchangeable: only word sense decides.
        if any(g.upper() in ambiguous_letters for g in glosses):
            return None
        word = "".join(glosses)
        # Simple title case: JUAN → Juan.
        if word.isupper():
            word = word.capitalize()
        return f"{word}."

    return None


def simple_lexical_spanish(glosses: Sequence[str]) -> Optional[str]:
    """
    A single lexical gloss (HOLA, GRACIAS, …) is already the Spanish word.
    No need for the Cloud Run GGUF (~seconds of CPU).
    """
    if len(glosses) != 1:
        return None
    g = str(glosses[0]).strip()
    if not g or classify_gloss(g) != "other":
        return None
    if g.upper() in LETTER_LOOKALIKES:
        return None
    if any(ch in g for ch in "/_"):
        return None
    word = g.capitalize() if g.isupper() else g
    return word + "."


class RepeatGate:
    """
    Decide whether a consecutively repeated gloss is accepted.

    - digit: always accept (a document number can be 1 1 2 2 3)
    - letter: up to max_letter_consecutive in a row (ANNA exists, ANNNA does not)
    - other: never twice in a row

    The `other` rule is equality, not time. There used to be a seconds-based
    dedup window, but it was unreachable: the inference cooldown (1 s) is at
    least as long as the window, so two consecutive glosses always arrived
    "late" and none were dropped. Also, two identical lexical signs in a row
    have no meaning in an utterance: if they appear, it is classifier bounce.
    To say HOLA again the utterance must close, and close calls reset().
    """

    def __init__(self, max_letter_consecutive: int = 2):
        self.max_letter_consecutive = max_letter_consecutive
        self.last_gloss: Optional[str] = None
        self.run_len: int = 0

    def reset(self) -> None:
        self.last_gloss = None
        self.run_len = 0

    def allow(self, gloss: str) -> bool:
        """
        Args:
            gloss: Already normalized (uppercase, ``EL/ELLA``, …).

        Returns:
            False if this is a forbidden repeat (3rd+ letter, duplicated lexical).
        """
        if self.last_gloss != gloss:
            self.last_gloss = gloss
            self.run_len = 1
            return True

        kind = classify_gloss(gloss)

        if kind == "digit":
            self.run_len += 1
            return True

        if kind == "letter":
            if self.run_len >= self.max_letter_consecutive:
                return False
            self.run_len += 1
            return True

        return False
