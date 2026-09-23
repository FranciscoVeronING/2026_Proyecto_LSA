"""Modo oyente: Chrome transcribe; acá se pide español → glosas al semántico remoto."""

from __future__ import annotations

from typing import Any

from semantic.remote import configured, ping


class HearingSession:

    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.semantic_ready = False
        self.semantic_error = ""
        self.spanish_text = ""
        self.gloss_text = ""
        try:
            if not configured():
                raise RuntimeError("Falta LSA_SEMANTIC_URL")
            if not ping():
                raise RuntimeError("El servidor semántico no responde")
            self.semantic_ready = True
            print("[backend] Modo oyente: voz → glosas (semántico remoto).")
        except Exception as exc:
            self.semantic_error = str(exc)
            self.semantic_ready = False
            print(f"[backend] Modo oyente sin traductor: {exc}")

    def capture_config(self) -> dict[str, Any]:
        return {"mode": "hearing"}

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "hearing",
            "classifier_ready": False,
            "semantic_ready": self.semantic_ready,
            "semantic_error": self.semantic_error,
            "spanish": self.spanish_text,
            "glosses": self.gloss_text,
            "device": "cpu",
        }

    def reset_session(self, left_handed: bool = False) -> None:
        self.spanish_text = ""
        self.gloss_text = ""
        return None

    def ingest_sign(self, frames: list) -> dict[str, Any]:
        raise RuntimeError("Modo oyente: no hay clasificación de señas.")

    def ingest_speech(self, spanish: str, final: bool = True) -> dict[str, Any]:
        text = (spanish or "").strip()
        if not text:
            return self.snapshot()
        self.spanish_text = text
        if not final:
            return {**self.snapshot(), "closed": False}
        glosses = text
        if self.semantic_ready:
            try:
                from semantic.remote import last_timing, translate_oyente
                import time

                t0 = time.perf_counter()
                glosses = translate_oyente(text) or text
                remote_ms = (time.perf_counter() - t0) * 1000
                timing = last_timing()
            except Exception as exc:
                print(f"[backend] Oyente remoto falló: {exc}")
                glosses = text
                remote_ms = 0.0
                timing = {}
        else:
            remote_ms = 0.0
            timing = {}
        self.gloss_text = glosses
        snap = self.snapshot()
        snap.update({"closed": True, "spanish": text, "glosses": glosses, "caption": glosses})
        from backend.metrics import log_event

        log_event(
            "hearing",
            spanish=text,
            glosses=glosses,
            remote_ms=round(remote_ms, 1),
            server_ms=timing.get("server_ms"),
        )
        return snap

    def note_activity(self) -> None:
        return None

    def close_utterance(self) -> dict[str, Any]:
        return self.snapshot()

    def clear_conversation(self) -> None:
        self.spanish_text = ""
        self.gloss_text = ""
        return None
