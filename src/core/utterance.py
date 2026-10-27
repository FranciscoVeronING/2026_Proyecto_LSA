"""Accumulate glosses until they form an utterance."""

import classifier.config as cfg
from core.repeat_policy import RepeatGate


def normalize_gloss(raw_name: str) -> str:
    """
    Classifier class name from the ``.pth`` → token the translator sees.

    ``GLOSS_NORMALIZER`` only holds **aliases** (``el_ella`` → ``EL``).
    Everything else is uppercased (``hola`` → ``HOLA``).
    """
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
    Collect accepted glosses. After `pause_sec` with no signing activity,
    close the utterance so the translator can take it.

    Whether a repeated gloss is accepted is delegated to RepeatGate.

    "Activity" is not the same as "accepted gloss". The countdown restarts on
    any recognized sign, even if RepeatGate drops it as a repeat, and when
    hands move in front of the camera. If only accepted glosses counted,
    repeating a sign would let the clock run and close the utterance mid-phrase.
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
        """
        Try to enqueue a classifier prediction.

        Args:
            gloss_raw: Raw class name (``el_ella``, ``A``, …).
            confidence: Top-1 softmax.
            now: Unix timestamp (seconds).

        Returns:
            True if the gloss landed in ``self.glosses``.
        """
        if confidence < self.min_confidence:
            return False
        gloss = normalize_gloss(gloss_raw)
        if not gloss or gloss == "DESCONOCIDO":
            return False

        accepted = self.repeat_gate.allow(gloss)
        # Recognizing a sign counts as activity even if it is dropped.
        self.last_activity_at = now
        if not accepted:
            return False
        self.glosses.append(gloss)
        return True

    def note_signing_activity(self, now):
        """Hands moving on camera: the person is still signing even if the
        classifier has not resolved anything yet."""
        if self.glosses:
            self.last_activity_at = now

    def maybe_close(self, now):
        """
        If ``pause_sec`` passed with no activity, return the glosses and empty the buffer.

        Returns:
            List of glosses, or None if it is not time to close.
        """
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
        """Current glosses joined with spaces, or an empty string."""
        return " ".join(self.glosses) if self.glosses else ""
