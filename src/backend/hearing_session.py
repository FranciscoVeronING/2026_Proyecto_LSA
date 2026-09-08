"""Sesión liviana del modo oyente: la extensión transcribe voz a subtítulos."""

from __future__ import annotations

from typing import Any


class HearingSession:
    """
    No carga clasificador ni LLM. El reconocimiento de voz corre en Chrome.
    Más adelante acá irá la LLM audio → glosas LSA.
    """

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.semantic_ready = True
        self.semantic_error = ""
        print("[backend] Modo oyente: voz → subtítulos (sin TinySkeleton ni GGUF).")

    def capture_config(self) -> dict[str, Any]:
        return {"mode": "hearing"}

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "hearing",
            "classifier_ready": False,
            "semantic_ready": True,
            "spanish": "",
            "glosses": [],
            "device": "cpu",
        }

    def reset_session(self, left_handed: bool = False) -> None:
        return None

    def ingest_sign(self, frames: list) -> dict[str, Any]:
        raise RuntimeError("Modo oyente: no hay clasificación de señas.")

    def note_activity(self) -> None:
        return None

    def close_utterance(self) -> dict[str, Any]:
        return self.snapshot()

    def clear_conversation(self) -> None:
        return None
