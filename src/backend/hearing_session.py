"""Hearing mode: Chrome transcribes; here Spanish → glosses on the remote semantic service."""

from __future__ import annotations

import time
from typing import Any

from core.enie import restore_enie_glosses
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
                raise RuntimeError("Falta la URL del traductor semántico")
            if not ping():
                raise RuntimeError("El traductor Cloud Run no responde")
            self.semantic_ready = True
            print("[backend] Hearing mode: voice → glosses (remote semantic).")
        except Exception as exc:
            self.semantic_error = str(exc)
            self.semantic_ready = False
            print(f"[backend] Hearing mode without translator: {exc}")

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
        raise RuntimeError("Modo oyente: no se clasifican señas.")

    def ingest_speech(self, spanish: str, final: bool = True) -> dict[str, Any]:
        text = (spanish or "").strip()
        if not text:
            return self.snapshot()
        self.spanish_text = text
        if not final:
            return {**self.snapshot(), "closed": False}
        glosses = ""
        remote_ms = 0.0
        timing = {}
        tokens = text.split()
        # A single word goes to T-pose as a gloss. A sentence must go through Cloud Run.
        if len(tokens) <= 1:
            glosses = tokens[0].strip().upper() if tokens else ""
            print(f"[backend] /hearing single word (no semantic): {text!r} → {glosses!r}")
        else:
            try:
                from semantic.remote import last_timing, ping, translate_hearing

                if not self.semantic_ready:
                    self.semantic_ready = ping()
                print(f"[backend] /hearing → Cloud Run POST /hearing: {text!r}")
                t0 = time.perf_counter()
                glosses = translate_hearing(text)
                remote_ms = (time.perf_counter() - t0) * 1000
                timing = last_timing()
                print(f"[backend] Cloud Run /hearing → glosses={glosses!r} {remote_ms:.0f}ms")
            except Exception as exc:
                print(f"[backend] Remote hearing failed: {exc}")
                self.semantic_error = str(exc)
                self.semantic_ready = False
        glosses = restore_enie_glosses(glosses, spanish=text)
        self.gloss_text = glosses
        clip = None
        delivery_ms = 0.0
        if glosses:
            try:
                from avatar.remote import clip_for_glosses

                t_del = time.perf_counter()
                clip = clip_for_glosses(glosses)
                delivery_ms = (time.perf_counter() - t_del) * 1000
            except Exception as exc:
                print(f"[backend] Remote T-pose failed: {exc}")
                clip = {"ok": False, "error": str(exc), "frames": [], "missing": []}
        queue_ms = timing.get("queue_ms") if timing.get("queue_ms") is not None else timing.get("queued_ms")
        infer_ms = timing.get("inference_ms") if timing.get("inference_ms") is not None else (
            timing.get("infer_ms") if timing.get("infer_ms") is not None else timing.get("server_ms")
        )
        transport = timing.get("transport_ms")
        if transport is None and remote_ms:
            from utils.benchmark_logger import transport_ms as net_ms

            transport = net_ms(timing.get("roundtrip_ms") or remote_ms, queue_ms, infer_ms)
        snap = self.snapshot()
        snap.update({
            "closed": True,
            "spanish": text,
            "glosses": glosses,
            "caption": glosses,
            "avatar": clip,
            "pause_ms": 0.0,
            "queue_ms": queue_ms,
            "inference_ms": infer_ms,
            "transport_ms": transport,
            "delivery_ms": round(delivery_ms, 1),
        })
        from backend.metrics import log_event, record_pipeline_turn

        log_event(
            "hearing",
            spanish=text,
            glosses=glosses,
            pause_ms=0.0,
            queued_ms=queue_ms,
            queue_ms=queue_ms,
            infer_ms=infer_ms,
            inference_ms=infer_ms,
            transport_ms=transport,
            remote_ms=round(remote_ms, 1),
            server_ms=timing.get("server_ms"),
            delivery_ms=round(delivery_ms, 1),
        )
        if glosses:
            record_pipeline_turn(
                "inverse",
                pause_ms=0,
                queue_ms=queue_ms,
                inference_ms=infer_ms,
                transport_ms=transport,
                delivery_ms=delivery_ms,
            )
        return snap

    def note_activity(self) -> None:
        return None

    def close_utterance(self, pause_ms: float | None = None) -> dict[str, Any]:
        return self.snapshot()
