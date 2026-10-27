"""Política de aceptación de glosas repetidas y formato de secuencias literales."""

from __future__ import annotations

from typing import Optional, Sequence


# Letras que el clasificador intercambia entre sí o con una seña léxica:
# I, T y OJO tienen casi la misma configuración de mano y solo se distinguen
# por dónde se apoya (mejilla / debajo de la boca / junto al ojo).
# Un deletreo que las contenga necesita el criterio del traductor.
AMBIGUOUS_LETTERS = frozenset({"I", "T"})

# Señas léxicas que el clasificador confunde con una letra (misma forma de mano).
# Si aparecen al lado o entre letras, casi seguro querían deletrear la letra.
LETTER_LOOKALIKES = {
    "OJO": "I",
    "VIERNES": "V",
    "AÑOS": "G",
}


def classify_gloss(gloss: str) -> str:
    """Clasifica una glosa normalizada: digit | letter | other."""
    if not gloss or len(gloss) != 1:
        return "other"
    if gloss.isdigit():
        return "digit"
    if gloss.isalpha():
        return "letter"
    return "other"


def resolve_letter_lookalikes(glosses: Sequence[str]) -> list[str]:
    """
    OJO / VIERNES / AÑOS pegadas a letras → I / V / G.

    Varias pasadas: ``M OJO VIERNES A`` primero cambia VIERNES (junto a A)
    y después OJO (junto a V). Un OJO o AÑOS sueltos se dejan como seña.
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
    Junta letras o dígitos **seguidos** antes de mandarlos al traductor.

    ``NOMBRE MIO P E P A`` → ``["NOMBRE", "MIO", "Pepa"]``
    ``EDAD TUYO 6 7`` → ``["EDAD", "TUYO", "67"]``

    Una letra o un dígito suelto se deja igual (puede ser glosa, no deletreo).
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
    Si el enunciado es solo deletreo o solo dígitos, arma el texto a mostrar/decir.
    Evita depender del traductor (que a veces responde vacío).

    Devuelve None cuando conviene que decida el traductor con el contexto.

    Nota sobre O/0 y 2/V: no necesitan excepción acá. Si la secuencia quedó toda
    de un tipo, el carácter ya es del tipo correcto; y si el clasificador eligió
    el tipo equivocado, la secuencia queda mixta y cae igual en el traductor.
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


def simple_lexical_spanish(glosses: Sequence[str]) -> Optional[str]:
    """
    Una sola glosa léxica (HOLA, GRACIAS, …) ya es la palabra en español.
    No hace falta el GGUF de Cloud Run (~segundos de CPU).
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
        """
        Args:
            gloss: Ya normalizada (mayúsculas, ``EL/ELLA``, …).

        Returns:
            False si es repetición prohibida (letra 3ª+, léxica duplicada).
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
