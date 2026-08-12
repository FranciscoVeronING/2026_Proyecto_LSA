"""
Pipeline de inferencia por participante.

Encapsula el estado que antes vivía como variables locales en main.py
y en shared_state global, para soportar múltiples usuarios en la webapp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

import classifier.config as cfg
from app.capture import (
    LandmarkSmoother,
    extract_normalized_vector_from_flat,
    prepare_input_tensor,
    should_start_recording,
)
from app.settings import SessionSettings
from app.utterance import UtteranceBuffer, normalize_gloss
from core.landmarks import compute_landmark_hand_motion


@dataclass
class PipelineEvent:
    type: str
    payload: dict = field(default_factory=dict)


@dataclass
class ParticipantState:
    top3: list = field(default_factory=list)
    prediction: str = "..."
    confidence: float = 0.0
    last_inference_time: float = 0.0
    utterance_glosses: list = field(default_factory=list)
    last_utterance: str = ""
    spanish_text: str = ""
    semantic_busy: bool = False
    buffer_len: int = 0
    still_frames: int = 0
    is_recording: bool = False


class ParticipantPipeline:
    """Un participante: captura de landmarks → buffer → cola de inferencia."""

    def __init__(
        self,
        participant_id: str,
        left_handed: bool = False,
        is_signer: bool = True,
        settings: Optional[SessionSettings] = None,
        device: Optional[torch.device] = None,
        on_enqueue: Optional[Callable[[str, torch.Tensor], None]] = None,
        on_utterance_closed: Optional[Callable[[str, list], None]] = None,
        landmarks_already_mirrored: bool = False,
    ):
        self.participant_id = participant_id
        self.left_handed = left_handed
        self.is_signer = is_signer
        self.settings = settings or SessionSettings()
        self.device = device or torch.device("cpu")
        self.on_enqueue = on_enqueue
        self.on_utterance_closed = on_utterance_closed
        # Si el navegador espejó la imagen antes de MediaPipe, no espejar el vector.
        self.landmarks_already_mirrored = landmarks_already_mirrored

        self.state = ParticipantState()
        self.smoother = LandmarkSmoother(alpha=0.6)
        self.utterance_buffer = UtteranceBuffer(
            pause_sec=self.settings.utterance_pause_sec,
            min_confidence=self.settings.confidence_threshold,
            max_letter_consecutive=self.settings.letter_max_consecutive,
        )

        self.frames_temp_buffer: list[np.ndarray] = []
        self.prev_hand_vector: Optional[np.ndarray] = None
        self.consecutive_still_frames = 0
        self.consecutive_hands_frames = 0
        self.missing_hands_frames = 0
        self.last_enqueue_time = 0.0
        self.last_seen_inference_time = 0.0

    def update_settings(self, settings: SessionSettings) -> None:
        self.settings = settings
        self.settings.apply_to_utterance_buffer(self.utterance_buffer)

    def _hands_present(self, left_hand: list, right_hand: list) -> bool:
        return bool(left_hand or right_hand)

    def _can_enqueue(self) -> bool:
        now = time.time()
        if now - self.last_enqueue_time < cfg.INFERENCE_COOLDOWN_SEC:
            return False
        if now - self.state.last_inference_time < cfg.INFERENCE_COOLDOWN_SEC:
            return False
        return True

    def _enqueue_buffer(self) -> bool:
        if not self._can_enqueue():
            return False
        tensor = prepare_input_tensor(self.frames_temp_buffer, self.device)
        if tensor is None:
            return False
        if self.on_enqueue:
            self.on_enqueue(self.participant_id, tensor)
        self.last_enqueue_time = time.time()
        return True

    def _reset_capture(self) -> None:
        self.frames_temp_buffer = []
        self.consecutive_still_frames = 0
        self.missing_hands_frames = 0
        self.smoother.reset()

    def process_landmarks(
        self,
        pose: list,
        left_hand: list,
        right_hand: list,
        motion_pixels: int = 0,
        timestamp: Optional[float] = None,
    ) -> list[PipelineEvent]:
        """Procesa un frame de landmarks crudos. Devuelve eventos para broadcast."""
        if not self.is_signer:
            return []

        now = timestamp or time.time()
        events: list[PipelineEvent] = []

        hands_present = self._hands_present(left_hand, right_hand)
        is_recording = len(self.frames_temp_buffer) > 0

        # Normalizar: si la imagen ya fue espejada en el cliente, no espejar el vector.
        apply_mirror = self.left_handed and not self.landmarks_already_mirrored
        current_vector = None
        landmark_motion_val = 0.0

        if hands_present:
            current_vector = extract_normalized_vector_from_flat(
                pose, left_hand, right_hand, left_handed=apply_mirror
            )
            landmark_motion_val = compute_landmark_hand_motion(
                current_vector, self.prev_hand_vector, cfg.POSE_DIM
            )
            self.prev_hand_vector = current_vector.copy()
            self.consecutive_hands_frames += 1
        else:
            self.consecutive_hands_frames = 0

        is_moving_pixels = motion_pixels > self.settings.motion_pixel_threshold
        is_moving = is_moving_pixels or landmark_motion_val > cfg.LANDMARK_MOTION_THRESHOLD

        if hands_present and landmark_motion_val > cfg.LANDMARK_MOTION_THRESHOLD:
            self.utterance_buffer.note_signing_activity(now)

        if not is_recording and should_start_recording(
            self.settings.capture_mode,
            hands_present,
            is_moving,
            self.consecutive_hands_frames,
            static_frames_to_start=self.settings.static_hands_frames_to_start,
        ):
            is_recording = True

        if is_recording:
            if hands_present and current_vector is not None:
                self.missing_hands_frames = 0
                self.frames_temp_buffer.append(self.smoother.update(current_vector))

                if is_moving:
                    self.consecutive_still_frames = 0
                else:
                    self.consecutive_still_frames += 1

                if (
                    len(self.frames_temp_buffer) >= cfg.CAPTURE_BUFFER_SIZE
                    or self.consecutive_still_frames >= self.settings.still_frames_limit
                ):
                    self._enqueue_buffer()
                    self._reset_capture()
            else:
                self.missing_hands_frames += 1
                if len(self.frames_temp_buffer) > 0:
                    self.frames_temp_buffer.append(self.frames_temp_buffer[-1])

                if self.missing_hands_frames >= cfg.MISSING_HANDS_LIMIT:
                    self._enqueue_buffer()
                    self._reset_capture()

        self.state.buffer_len = len(self.frames_temp_buffer)
        self.state.still_frames = self.consecutive_still_frames
        self.state.is_recording = is_recording

        closed = self.utterance_buffer.maybe_close(now)
        if closed is not None:
            self.state.utterance_glosses = []
            self.state.last_utterance = " ".join(closed)
            if self.on_utterance_closed:
                self.on_utterance_closed(self.participant_id, closed)
            events.append(
                PipelineEvent(
                    type="utterance_pending",
                    payload={"glosses": closed, "joined": " ".join(closed)},
                )
            )

        return events

    def apply_inference_result(
        self, top3: list, timestamp: Optional[float] = None
    ) -> list[PipelineEvent]:
        """Aplica el resultado del clasificador y devuelve eventos."""
        events: list[PipelineEvent] = []
        now = timestamp or time.time()

        self.state.top3 = top3
        if top3:
            self.state.prediction = top3[0][0].upper()
            self.state.confidence = top3[0][1]
        self.state.last_inference_time = now

        events.append(
            PipelineEvent(
                type="top3",
                payload={
                    "top3": [{"name": n, "confidence": c} for n, c in top3],
                    "prediction": self.state.prediction,
                    "confidence": self.state.confidence,
                },
            )
        )

        if top3 and now > self.last_seen_inference_time:
            self.last_seen_inference_time = now
            gloss_name, gloss_conf = top3[0][0], top3[0][1]
            if self.utterance_buffer.try_add(gloss_name, gloss_conf, now):
                self.state.utterance_glosses = list(self.utterance_buffer.glosses)
                events.append(
                    PipelineEvent(
                        type="gloss_added",
                        payload={
                            "gloss": normalize_gloss(gloss_name),
                            "pending": self.utterance_buffer.pending_text(),
                            "glosses": list(self.utterance_buffer.glosses),
                        },
                    )
                )

        return events

    def apply_semantic_result(
        self, text: str, glosses: str, busy: bool = False
    ) -> PipelineEvent:
        self.state.spanish_text = text
        self.state.semantic_busy = busy
        self.state.utterance_glosses = []
        return PipelineEvent(
            type="utterance_closed",
            payload={"spanish": text, "glosses": glosses},
        )

    def get_snapshot(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "is_signer": self.is_signer,
            "left_handed": self.left_handed,
            "top3": [{"name": n, "confidence": c} for n, c in self.state.top3],
            "prediction": self.state.prediction,
            "confidence": self.state.confidence,
            "pending_glosses": self.utterance_buffer.pending_text(),
            "glosses": list(self.state.utterance_glosses),
            "spanish_text": self.state.spanish_text,
            "semantic_busy": self.state.semantic_busy,
            "buffer_len": self.state.buffer_len,
            "still_frames": self.state.still_frames,
            "is_recording": self.state.is_recording,
            "settings": self.settings.to_dict(),
        }
