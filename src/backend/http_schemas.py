"""Frame/point cap on POST /sign. The cropper never sends more than 60."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# The extension cropper never sends more than CAPTURE_BUFFER_SIZE (60).
MAX_SIGN_FRAMES = 60
MAX_POINTS_PER_LIST = 33


class SessionIn(BaseModel):
    left_handed: bool = False


class HearingIn(BaseModel):
    spanish: str = ""
    final: bool = True


class UtteranceEndIn(BaseModel):
    pause_ms: float | None = None


class ClientMetricsIn(BaseModel):
    channel: str = "direct"
    delivery_ms: float | None = None
    pause_ms: float | None = None
    fps: float | None = None
    holistic_fps: float | None = None
    rss_mb: float | None = None


class SttIn(BaseModel):
    wav_b64: str = ""
    lang: str = "es-ES"


class TtsIn(BaseModel):
    text: str = Field(default="", max_length=500)


class SignIn(BaseModel):

    frames: list[dict] = Field(default_factory=list, max_length=MAX_SIGN_FRAMES)
    client: dict = Field(default_factory=dict)

    @field_validator("frames")
    @classmethod
    def frames_are_objects(cls, value: list) -> list:
        if not isinstance(value, list):
            raise ValueError("frames must be a list")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("each frame must be a JSON object")
            for key in ("pose", "left_hand", "right_hand"):
                pts = item.get(key)
                if pts is None:
                    continue
                if not isinstance(pts, list):
                    raise ValueError(f"{key} must be a list or null")
                if len(pts) > MAX_POINTS_PER_LIST:
                    raise ValueError(f"{key} has too many points")
        return value
