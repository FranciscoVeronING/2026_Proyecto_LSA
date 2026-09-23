"""
Sesión de interpretación: clasificador + buffer de glosas + traductor.

La extensión recorta señas (Holistic + capture.js) y manda el fin de
enunciado. Acá no se extraen landmarks: se clasifica, se acumula y se traduce.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if getattr(sys, "frozen", False):
    _root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal"))
    for _lib in (_root / "torch" / "lib", _root):
        if _lib.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_lib))
            except OSError:
                pass

import torch

import classifier.config as cfg
from app.utterance import UtteranceBuffer, normalize_gloss
from backend.landmarks_payload import LandmarkSmoother, frames_to_matrix, vector_from_frame
from classifier.arch import load_classifier_bundle
from core.conversation_memory import ConversationMemory
from core.repeat_policy import format_literal_utterance
from semantic.config import CONVERSATION_HISTORY_SIZE, DEFAULT_MODEL_ID, USE_CONVERSATION_HISTORY


class LSASession:
    def __init__(self, enable_llm: bool = True, semantic_model_id: str | None = None):
        # La LLM en un hilo aparte: si carga acá, /health se queda mudo un rato.
        self.lock = threading.RLock()
        self._llm_gate = threading.Lock()
        self.enable_llm = enable_llm
        self._wanted_model_id = semantic_model_id or DEFAULT_MODEL_ID
        self.left_handed = False
        self.device = torch.device("cpu")
        print("[backend] Clasificador en CPU. Traductor: servidor remoto.")
        self.model, self.idx_to_class = load_classifier_bundle(self.device)
        print(f"[backend] Clasificador: {len(self.idx_to_class)} señas en {self.device}")
        self.buffer = UtteranceBuffer(
            pause_sec=cfg.UTTERANCE_PAUSE_SEC,
            min_confidence=cfg.CONFIDENCE_THRESHOLD,
            max_letter_consecutive=cfg.LETTER_MAX_CONSECUTIVE,
        )
        self.memory = ConversationMemory(maxlen=CONVERSATION_HISTORY_SIZE)
        self.last_enqueue_time = 0.0
        self.last_inference_time = 0.0
        self.top3: list[tuple[str, float]] = []
        self.spanish_text = ""
        self.last_utterance = ""
        self.semantic_busy = False
        self.semantic_ready = False
        self.semantic_error = ""
        self._translate_glosses = None
        self._current_model_id = ""
        if enable_llm:
            threading.Thread(target=self._bootstrap_llm, daemon=True).start()

    def _bootstrap_llm(self):
        try:
            from semantic.remote import configured, ping, translate_sordo

            if not configured():
                raise RuntimeError("Falta LSA_SEMANTIC_URL")
            if not ping():
                raise RuntimeError("El servidor semántico no responde (túnel Cloudflare / run_semantic_server.py)")
            with self._llm_gate:
                with self.lock:
                    self._translate_glosses = translate_sordo
                    self._current_model_id = "remote"
                    self.semantic_ready = True
                    self.semantic_error = ""
            print("[backend] Traductor remoto listo.")
        except Exception as e:
            with self.lock:
                self.semantic_error = str(e)
                self.semantic_ready = False
            print(f"[backend] LLM remota no disponible: {e}")

    def switch_semantic_model(self, model_id: str) -> None:
        return None

    def reset_session(self, left_handed: bool):
        with self.lock:
            self.left_handed = bool(left_handed)
            self.buffer = UtteranceBuffer(
                pause_sec=cfg.UTTERANCE_PAUSE_SEC,
                min_confidence=cfg.CONFIDENCE_THRESHOLD,
                max_letter_consecutive=cfg.LETTER_MAX_CONSECUTIVE,
            )
            self.memory.reset_session()
            self.top3 = []
            self.spanish_text = ""
            self.last_utterance = ""
            self.semantic_busy = False
            self.last_enqueue_time = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "left_handed": self.left_handed,
                "classifier_ready": self.model is not None,
                "num_classes": len(self.idx_to_class),
                "semantic_ready": self.semantic_ready,
                "semantic_error": self.semantic_error,
                "semantic_model": self._current_model_id,
                "semantic_busy": self.semantic_busy,
                "top3": [{"gloss": n, "confidence": c} for n, c in self.top3],
                "glosses": list(self.buffer.glosses),
                "pending_text": self.buffer.pending_text(),
                "spanish": self.spanish_text,
                "last_utterance": self.last_utterance,
                "conversation_turns": len(self.memory.turns),
                "device": str(self.device),
            }

    def note_activity(self):
        with self.lock:
            self.buffer.note_signing_activity(time.time())

    def ingest_sign(self, frames: list[dict], client: dict | None = None) -> dict[str, Any]:
        # La seña ya viene recortada. Acá: EMA → (16, 225) → TinySkeleton.
        t0 = time.perf_counter()
        now = time.time()
        client = client or {}
        n = len(frames or [])
        with_pose = sum(1 for f in frames or [] if f.get("pose"))
        with_lh = sum(1 for f in frames or [] if f.get("left_hand"))
        with_rh = sum(1 for f in frames or [] if f.get("right_hand"))
        print(
            f"[backend] /sign recibido: {n} frames "
            f"(pose={with_pose} mano_izq={with_lh} mano_der={with_rh})"
        )

        with self.lock:
            cool = cfg.INFERENCE_COOLDOWN_SEC
            if now - self.last_enqueue_time < cool or now - self.last_inference_time < cool:
                print("[backend] /sign rechazado: cooldown")
                from backend.metrics import log_event

                log_event(
                    "sign",
                    accepted=False,
                    reason="cooldown",
                    n_frames=n,
                    capture_ms=client.get("capture_ms"),
                    holistic_ms=client.get("holistic_ms"),
                )
                return {"accepted": False, "reason": "cooldown", **self.snapshot()}
            if n < cfg.MIN_CAPTURE_FRAMES:
                print(
                    f"[backend] /sign rechazado: too_short "
                    f"({n} < min {cfg.MIN_CAPTURE_FRAMES})"
                )
                from backend.metrics import log_event

                log_event(
                    "sign",
                    accepted=False,
                    reason="too_short",
                    n_frames=n,
                    capture_ms=client.get("capture_ms"),
                    holistic_ms=client.get("holistic_ms"),
                )
                return {"accepted": False, "reason": "too_short", **self.snapshot()}

        smoother = LandmarkSmoother(alpha=0.6)
        vectors = []
        for frame in frames:
            vec = vector_from_frame(frame, left_handed=self.left_handed)
            vectors.append(smoother.update(vec))

        matrix = frames_to_matrix(vectors)
        print(
            f"[backend] /sign tensor: entrada={n} "
            f"tras muestreo={matrix.shape} (esperado {cfg.MAX_FRAMES}x{matrix.shape[1] if matrix.ndim == 2 else '?'})"
        )
        if matrix.shape[0] != cfg.MAX_FRAMES:
            print("[backend] /sign rechazado: bad_tensor")
            from backend.metrics import log_event

            log_event("sign", accepted=False, reason="bad_tensor", n_frames=n)
            return {"accepted": False, "reason": "bad_tensor", **self.snapshot()}
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_inf = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
            top3 = []
            for conf, idx in zip(values.tolist(), indices.tolist()):
                name = self.idx_to_class.get(idx, "desconocido")
                top3.append((name, float(conf)))
        infer_ms = (time.perf_counter() - t_inf) * 1000

        added = False
        activity = False
        gloss = normalize_gloss(top3[0][0]) if top3 else ""
        with self.lock:
            self.last_enqueue_time = now
            self.last_inference_time = time.time()
            self.top3 = top3
            if top3:
                prev_act = self.buffer.last_activity_at
                added = self.buffer.try_add(top3[0][0], top3[0][1], self.last_inference_time)
                activity = self.buffer.last_activity_at != prev_act

        print(
            "[backend] /sign top3: "
            + " | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3)
            + (f" → agregada {gloss}" if added else " → no se agregó (repetición/umbral)")
        )
        snap = self.snapshot()
        total_ms = (time.perf_counter() - t0) * 1000
        snap.update({
            "accepted": True,
            "added": added,
            "activity": activity,
            "gloss": top3[0][0] if top3 else None,
            "ms": round(total_ms, 1),
        })
        from backend.metrics import log_event

        log_event(
            "sign",
            accepted=True,
            added=added,
            n_frames=n,
            gloss=top3[0][0] if top3 else "",
            conf1=top3[0][1] if top3 else None,
            conf2=top3[1][1] if len(top3) > 1 else None,
            classifier_ms=round(infer_ms, 1),
            total_ms=round(total_ms, 1),
            capture_ms=client.get("capture_ms"),
            holistic_ms=client.get("holistic_ms"),
            n_raw=client.get("n_raw"),
        )
        print(
            f"[backend] /sign respuesta: accepted=True added={added} "
            f"glosa={snap.get('gloss')} pending={snap.get('pending_text')!r}"
        )
        return snap

    def close_utterance(self) -> dict[str, Any]:
        with self.lock:
            closed = list(self.buffer.glosses)
            self.buffer.glosses.clear()
            self.buffer.repeat_gate.reset()
            self.buffer.last_activity_at = None
            if not closed:
                print("[backend] /utterance/end: no había glosas pendientes")
                snap = self.snapshot()
                snap.update({"closed": False, "glosses": [], "spanish": self.spanish_text})
                return snap
            self.semantic_busy = True
            self.last_utterance = " ".join(closed)

        print(f"[backend] /utterance/end recibido: {closed!r}")
        t0 = time.perf_counter()
        text, via, remote_ms, server_ms = self._translate(closed)
        total_ms = (time.perf_counter() - t0) * 1000

        with self.lock:
            self.spanish_text = text
            self.semantic_busy = False
            self.memory.add_signer(text, glosses=" ".join(closed))

        snap = self.snapshot()
        snap.update({
            "closed": True,
            "glosses": closed,
            "spanish": text,
            "via": via,
            "ms": round(total_ms, 1),
        })
        from backend.metrics import log_event

        log_event(
            "utterance",
            glosses=" ".join(closed),
            spanish=text,
            via=via,
            remote_ms=round(remote_ms, 1),
            server_ms=round(server_ms, 1) if server_ms is not None else None,
            total_ms=round(total_ms, 1),
            n_glosses=len(closed),
        )
        print(f"[backend] /utterance/end respuesta: español={text!r} via={via} {total_ms:.0f}ms")
        return snap

    def _translate(self, glosses: list[str]) -> tuple[str, str, float, float | None]:
        # Letras/dígitos van literales. Sin traductor remoto, devolvemos las glosas.
        joined = " ".join(glosses)
        literal = format_literal_utterance(glosses)
        if literal is not None:
            print(f"[backend] Enunciado → literal: {joined} => {literal}")
            return literal, "literal", 0.0, None
        print(f"[backend] Enunciado → LLM: {joined}")
        with self.lock:
            fn = self._translate_glosses
        if fn is None:
            return joined, "fallback", 0.0, None
        try:
            from semantic.remote import last_timing

            history = self.memory.as_messages() if USE_CONVERSATION_HISTORY else None
            t0 = time.perf_counter()
            text = fn(joined, history_messages=history) or joined
            remote_ms = (time.perf_counter() - t0) * 1000
            timing = last_timing()
            return text, "llm", remote_ms, timing.get("server_ms")
        except Exception as e:
            print(f"[backend] Error LLM: {e}")
            return joined, "fallback", 0.0, None

    def clear_conversation(self):
        with self.lock:
            self.memory.clear()

    def capture_config(self) -> dict[str, Any]:
        return {
            "max_frames": cfg.MAX_FRAMES,
            "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
            "inference_cooldown_sec": cfg.INFERENCE_COOLDOWN_SEC,
            "motion_pixel_threshold": cfg.MOTION_PIXEL_THRESHOLD,
            "landmark_motion_threshold": cfg.LANDMARK_MOTION_THRESHOLD,
            "static_hands_frames_to_start": cfg.STATIC_HANDS_FRAMES_TO_START,
            "hands_frames_to_start": cfg.HANDS_FRAMES_TO_START,
            "still_frames_limit": cfg.STILL_FRAMES_LIMIT,
            "capture_buffer_size": cfg.CAPTURE_BUFFER_SIZE,
            "missing_hands_limit": cfg.MISSING_HANDS_LIMIT,
            "min_capture_frames": cfg.MIN_CAPTURE_FRAMES,
            "capture_mode": cfg.CAPTURE_MODE,
            "num_classes": len(self.idx_to_class),
            "utterance_pause_sec": cfg.UTTERANCE_PAUSE_SEC,
            "pose_dim": cfg.POSE_DIM,
            "mode": "signer",
        }
