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
    Junta glosas aceptadas. Cuando pasan `pause_sec` sin una nueva, cierra el
    enunciado para que lo tome el traductor.

    La decisión de aceptar una glosa repetida se delega en RepeatGate.
    """

    def __init__(
        self,
        pause_sec: float,
        dedup_sec: float,
        min_confidence: float,
        max_letter_consecutive: int = 2,
    ):
        self.pause_sec = pause_sec
        self.min_confidence = min_confidence
        self.glosses = []
        self.repeat_gate = RepeatGate(
            dedup_sec=dedup_sec,
            max_letter_consecutive=max_letter_consecutive,
        )

    def try_add(self, gloss_raw, confidence, now):
        if confidence < self.min_confidence:
            return False
        gloss = normalize_gloss(gloss_raw)
        if not gloss or gloss == "DESCONOCIDO":
            return False
        if not self.repeat_gate.allow(gloss, now):
            return False
        self.glosses.append(gloss)
        return True

    def maybe_close(self, now):
        if not self.glosses or self.repeat_gate.last_at is None:
            return None
        if (now - self.repeat_gate.last_at) < self.pause_sec:
            return None
        closed = list(self.glosses)
        self.glosses.clear()
        self.repeat_gate.reset()
        return closed

    def pending_text(self) -> str:
        return " ".join(self.glosses) if self.glosses else ""
