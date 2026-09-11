"""
Sesión de interpretación: clasificador + buffer de glosas + traductor.

La extensión recorta señas (Holistic + capture.js) y manda el fin de
enunciado. Acá no se extraen landmarks: se clasifica, se acumula y se traduce.
"""

from __future__ import annotations

import os
import json
import threading
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

import classifier.config as cfg
from app.utterance import UtteranceBuffer, normalize_gloss
from backend.landmarks_payload import LandmarkSmoother, frames_to_matrix, vector_from_frame
from classifier.arch import TinySkeletonClassifier
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
        print("[backend] Clasificador y traductor en CPU.")
        self.idx_to_class = self._load_classes()
        self.model = self._load_classifier()
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

    @staticmethod
    def _load_classes() -> dict:
        with open(cfg.CLASSES_PATH, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        return {v: k for k, v in class_to_idx.items()}

    def _load_classifier(self):
        model = TinySkeletonClassifier(
            cfg.FRAME_FEATURES_DIM,
            cfg.HIDDEN_DIM,
            num_heads=cfg.NUM_HEADS,
            num_layers=cfg.NUM_LAYERS,
            num_classes=len(self.idx_to_class),
            dropout_rate=cfg.DROPOUT_RATE,
        ).to(self.device)
        model.load_state_dict(
            torch.load(cfg.WEIGHTS_PATH, map_location=self.device, weights_only=True)
        )
        model.eval()
        print(f"[backend] Clasificador en {self.device}")
        return model

    def _bootstrap_llm(self):
        try:
            from semantic.translator import (
                get_active_model_id,
                load_model_and_tokenizer,
                translate_glosses,
            )

            with self._llm_gate:
                load_model_and_tokenizer(self._wanted_model_id)
                with self.lock:
                    self._translate_glosses = translate_glosses
                    self._current_model_id = get_active_model_id() or self._wanted_model_id
                    self.semantic_ready = True
                    self.semantic_error = ""
            print(f"[backend] Traductor semántico listo ({self._current_model_id}).")
        except Exception as e:
            with self.lock:
                self.semantic_error = str(e)
                self.semantic_ready = False
            print(f"[backend] LLM no disponible: {e}")

    def switch_semantic_model(self, model_id: str) -> None:
        # Recarga el GGUF en background; FastAPI y Tk siguen respondiendo.
        if not self.enable_llm or not model_id:
            return
        self._wanted_model_id = model_id
        threading.Thread(
            target=self._apply_semantic_switch,
            args=(model_id,),
            daemon=True,
            name="ilsa-semantic-switch",
        ).start()

    def _apply_semantic_switch(self, model_id: str) -> None:
        from semantic.translator import get_active_model_id, switch_model, translate_glosses

        with self.lock:
            if self._current_model_id == model_id and self.semantic_ready:
                return
            self.semantic_ready = False
            self.semantic_busy = True
            self.semantic_error = ""
        print(f"[backend] Cambio de traductor pedido: {model_id}")
        try:
            with self._llm_gate:
                wanted = self._wanted_model_id or model_id
                with self.lock:
                    if self._current_model_id == wanted and self.semantic_ready:
                        self.semantic_busy = False
                        return
                switch_model(wanted)
                with self.lock:
                    self._translate_glosses = translate_glosses
                    self._current_model_id = get_active_model_id() or wanted
                    self.semantic_ready = True
                    self.semantic_busy = False
                    self.semantic_error = ""
            print(f"[backend] Traductor semántico listo ({self._current_model_id}).")
        except Exception as e:
            with self.lock:
                self.semantic_error = str(e)
                self.semantic_ready = False
                self.semantic_busy = False
            print(f"[backend] No se pudo cargar {model_id}: {e}")

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

    def ingest_sign(self, frames: list[dict]) -> dict[str, Any]:
        # La seña ya viene recortada. Acá: EMA → (16, 225) → TinySkeleton.
        now = time.time()
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
                return {"accepted": False, "reason": "cooldown", **self.snapshot()}
            if n < cfg.MIN_CAPTURE_FRAMES:
                print(
                    f"[backend] /sign rechazado: too_short "
                    f"({n} < min {cfg.MIN_CAPTURE_FRAMES})"
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
            return {"accepted": False, "reason": "bad_tensor", **self.snapshot()}
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
            top3 = []
            for conf, idx in zip(values.tolist(), indices.tolist()):
                name = self.idx_to_class.get(idx, "desconocido")
                top3.append((name, float(conf)))

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
        snap.update({
            "accepted": True,
            "added": added,
            "activity": activity,
            "gloss": top3[0][0] if top3 else None,
        })
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
        text = self._translate(closed)

        with self.lock:
            self.spanish_text = text
            self.semantic_busy = False
            self.memory.add_signer(text, glosses=" ".join(closed))

        snap = self.snapshot()
        snap.update({"closed": True, "glosses": closed, "spanish": text})
        print(f"[backend] /utterance/end respuesta: español={text!r}")
        return snap

    def _translate(self, glosses: list[str]) -> str:
        # Letras/dígitos van literales. Si el GGUF no cargó, devolvemos las glosas.
        joined = " ".join(glosses)
        literal = format_literal_utterance(glosses)
        if literal is not None:
            print(f"[backend] Enunciado → literal: {joined} => {literal}")
            return literal
        print(f"[backend] Enunciado → LLM: {joined}")
        with self.lock:
            fn = self._translate_glosses
        if fn is None:
            return joined
        try:
            history = self.memory.as_messages() if USE_CONVERSATION_HISTORY else None
            return fn(joined, history_messages=history) or joined
        except Exception as e:
            print(f"[backend] Error LLM: {e}")
            return joined

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
            "utterance_pause_sec": cfg.UTTERANCE_PAUSE_SEC,
            "pose_dim": cfg.POSE_DIM,
            "mode": "signer",
        }
