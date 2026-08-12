"""Configuración por sesión/participante (no muta el módulo cfg global)."""

from dataclasses import dataclass, field

import classifier.config as cfg


@dataclass
class SessionSettings:
    confidence_threshold: float = cfg.CONFIDENCE_THRESHOLD
    motion_pixel_threshold: int = cfg.MOTION_PIXEL_THRESHOLD
    still_frames_limit: int = cfg.STILL_FRAMES_LIMIT
    static_hands_frames_to_start: int = cfg.STATIC_HANDS_FRAMES_TO_START
    capture_mode: str = cfg.CAPTURE_MODE  # auto | static | dynamic
    utterance_pause_sec: float = cfg.UTTERANCE_PAUSE_SEC
    letter_max_consecutive: int = cfg.LETTER_MAX_CONSECUTIVE
    voice_enabled: bool = cfg.VOICE
    show_landmarks: bool = False

    def apply_to_utterance_buffer(self, buffer) -> None:
        buffer.min_confidence = self.confidence_threshold
        buffer.pause_sec = self.utterance_pause_sec
        buffer.repeat_gate.max_letter_consecutive = self.letter_max_consecutive

    def to_dict(self) -> dict:
        return {
            "confidence_threshold": self.confidence_threshold,
            "motion_pixel_threshold": self.motion_pixel_threshold,
            "still_frames_limit": self.still_frames_limit,
            "static_hands_frames_to_start": self.static_hands_frames_to_start,
            "capture_mode": self.capture_mode,
            "utterance_pause_sec": self.utterance_pause_sec,
            "letter_max_consecutive": self.letter_max_consecutive,
            "voice_enabled": self.voice_enabled,
            "show_landmarks": self.show_landmarks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSettings":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
