"""Política de aceptación de glosas repetidas y formato de secuencias literales."""

from __future__ import annotations

from typing import Optional, Sequence


# Letras que el clasificador intercambia entre sí o con una seña léxica:
# I, T y OJO tienen casi la misma configuración de mano y solo se distinguen
# por dónde se apoya (mejilla / debajo de la boca / junto al ojo).
# Un deletreo que las contenga necesita el criterio de la LLM.
AMBIGUOUS_LETTERS = frozenset({"I", "T"})


def classify_gloss(gloss: str) -> str:
    """Clasifica una glosa normalizada: digit | letter | other."""
    if not gloss or len(gloss) != 1:
        return "other"
    if gloss.isdigit():
        return "digit"
    if gloss.isalpha():
        return "letter"
    return "other"


def format_literal_utterance(
    glosses: Sequence[str],
    ambiguous_letters: frozenset = AMBIGUOUS_LETTERS,
) -> Optional[str]:
    """
    Si el enunciado es solo deletreo o solo dígitos, arma el texto a mostrar/decir.
    Evita depender de la LLM (que a veces responde vacío).

    Devuelve None cuando conviene que decida la LLM con el contexto.

    Nota sobre O/0 y 2/V: no necesitan excepción acá. Si la secuencia quedó toda
    de un tipo, el carácter ya es del tipo correcto; y si el clasificador eligió
    el tipo equivocado, la secuencia queda mixta y cae igual en la LLM.
    """
    if not glosses:
        return None

    kinds = [classify_gloss(g) for g in glosses]

    if all(k == "digit" for k in kinds):
        return f"{''.join(glosses)}."

    if all(k == "letter" for k in kinds):
        # Una letra sola puede ser en realidad una seña léxica mal clasificada.
        if len(glosses) == 1:
            return None
        # I y T son intercambiables entre sí: solo el sentido de la palabra decide.
        if any(g.upper() in ambiguous_letters for g in glosses):
            return None
        word = "".join(glosses)
        # Título simple: JUAN → Juan.
        if word.isupper():
            word = word.capitalize()
        return f"{word}."

    return None


class RepeatGate:
    """
    Decide si una glosa consecutivamente repetida se acepta.

    - digit: siempre aceptar (un documento puede ser 1 1 2 2 3)
    - letter: hasta max_letter_consecutive consecutivas (existe ANNA, no ANNNA)
    - other: nunca dos veces seguidas

    La regla de `other` es por igualdad, no por tiempo. Antes había una ventana
    de dedup en segundos, pero era inalcanzable: el cooldown entre inferencias
    (1 s) es igual o mayor que la ventana, así que dos glosas consecutivas
    siempre llegaban "tarde" y ninguna se descartaba. Además, dos señas léxicas
    idénticas pegadas no tienen significado propio en un enunciado: si aparecen,
    es rebote del clasificador. Para volver a decir HOLA hay que cerrar el
    enunciado, y al cerrarse se llama a reset().
    """

    def __init__(self, max_letter_consecutive: int = 2):
        self.max_letter_consecutive = max_letter_consecutive
        self.last_gloss: Optional[str] = None
        self.run_len: int = 0

    def reset(self) -> None:
        self.last_gloss = None
        self.run_len = 0

    def allow(self, gloss: str) -> bool:
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
