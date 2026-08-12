"""Modelos Pydantic para mensajes HTTP y WebSocket."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CreateRoomResponse(BaseModel):
    room_id: str


class RoomStatusResponse(BaseModel):
    room_id: str
    participants: int
    classifier_ready: bool
    semantic_ready: bool


class JoinPayload(BaseModel):
    type: Literal["join"] = "join"
    name: str
    is_signer: bool
    left_handed: bool = False
    landmarks_already_mirrored: bool = True


class LandmarksPayload(BaseModel):
    type: Literal["landmarks"] = "landmarks"
    pose: list[float] = Field(default_factory=list)
    left_hand: list[float] = Field(default_factory=list)
    right_hand: list[float] = Field(default_factory=list)
    motion_pixels: int = 0
    mirrored: bool = False


class FramePayload(BaseModel):
    type: Literal["frame"] = "frame"
    image_b64: str
    motion_pixels: int = 0
    mirrored: bool = False


class ChatPayload(BaseModel):
    type: Literal["chat"] = "chat"
    text: str
    source: Literal["typed", "stt"] = "typed"


class SettingsPayload(BaseModel):
    type: Literal["settings"] = "settings"
    settings: dict[str, Any]


class SignalPayload(BaseModel):
    type: Literal["signal"] = "signal"
    signal_type: Literal["offer", "answer", "ice"]
    data: dict[str, Any]


class ClearContextPayload(BaseModel):
    type: Literal["clear_context"] = "clear_context"


class ServerMessage(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Top3Item(BaseModel):
    name: str
    confidence: float
