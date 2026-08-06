"""Acumulación de glosas hasta formar un enunciado."""

import classifier.config as cfg
from core.repeat_policy import RepeatGate


def normalize_gloss(raw_name: str) -> str:
    """Clave del clasificador → glosa normalizada en mayúsculas."""
    key = (raw_name or "").strip()
    mapped = cfg.GLOSS_NORMALIZER.get(key)
    if mapped:
        return mapped
    mapped = cfg.GLOSS_NORMALIZER.get(key.lower())
    if mapped:
        return mapped
    return key.upper()


class UtteranceBuffer:
    """
    Junta glosas aceptadas. Cuando pasan `pause_sec` sin actividad de señado,
    cierra el enunciado para que lo tome el traductor.

    La decisión de aceptar una glosa repetida se delega en RepeatGate.

    "Actividad" no es lo mismo que "glosa aceptada". La cuenta regresiva se
    reinicia con cualquier seña reconocida, incluso si RepeatGate la descarta
    por repetida, y con las manos moviéndose frente a la cámara. Si solo
    contara las aceptadas, repetir una seña dejaría correr el reloj y el
    enunciado podría cerrarse en medio de la frase.
    """

    def __init__(
        self,
        pause_sec: float,
        min_confidence: float,
        max_letter_consecutive: int = 2,
    ):
        self.pause_sec = pause_sec
        self.min_confidence = min_confidence
        self.glosses = []
        self.last_activity_at = None
        self.repeat_gate = RepeatGate(max_letter_consecutive=max_letter_consecutive)

    def try_add(self, gloss_raw, confidence, now):
        if confidence < self.min_confidence:
            return False
        gloss = normalize_gloss(gloss_raw)
        if not gloss or gloss == "DESCONOCIDO":
            return False

        accepted = self.repeat_gate.allow(gloss)
        # Reconocer una seña cuenta como actividad aunque se descarte.
        self.last_activity_at = now
        if not accepted:
            return False
        self.glosses.append(gloss)
        return True

    def note_signing_activity(self, now):
        """Manos moviéndose en cámara: la persona sigue señando aunque el
        clasificador todavía no haya resuelto nada."""
        if self.glosses:
            self.last_activity_at = now

    def maybe_close(self, now):
        if not self.glosses or self.last_activity_at is None:
            return None
        if (now - self.last_activity_at) < self.pause_sec:
            return None
        closed = list(self.glosses)
        self.glosses.clear()
        self.repeat_gate.reset()
        self.last_activity_at = None
        return closed

    def pending_text(self) -> str:
        return " ".join(self.glosses) if self.glosses else ""
