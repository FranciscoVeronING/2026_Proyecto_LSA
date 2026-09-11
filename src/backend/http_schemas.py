"""Tope de frames/puntos en POST /sign. El recortador no manda más de 60."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# El recortador de la extensión no manda más de CAPTURE_BUFFER_SIZE (60).
MAX_SIGN_FRAMES = 60
MAX_POINTS_PER_LIST = 33


class SessionIn(BaseModel):
    left_handed: bool = False


class SignIn(BaseModel):

    frames: list[dict] = Field(default_factory=list, max_length=MAX_SIGN_FRAMES)

    @field_validator("frames")
    @classmethod
    def frames_are_objects(cls, value: list) -> list:
        if not isinstance(value, list):
            raise ValueError("frames debe ser una lista")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("cada frame debe ser un objeto JSON")
            for key in ("pose", "left_hand", "right_hand"):
                pts = item.get(key)
                if pts is None:
                    continue
                if not isinstance(pts, list):
                    raise ValueError(f"{key} debe ser lista o null")
                if len(pts) > MAX_POINTS_PER_LIST:
                    raise ValueError(f"{key} tiene demasiados puntos")
        return value
