"""Gestión de salas y participantes."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket

from app.pipeline import ParticipantPipeline
from app.settings import SessionSettings


@dataclass
class Participant:
    id: str
    name: str
    is_signer: bool
    left_handed: bool
    websocket: WebSocket
    pipeline: Optional[ParticipantPipeline] = None
    landmarks_already_mirrored: bool = True


@dataclass
class Room:
    id: str
    participants: dict[str, Participant] = field(default_factory=dict)
    max_participants: int = 2

    def is_full(self) -> bool:
        return len(self.participants) >= self.max_participants

    def other_participant(self, participant_id: str) -> Optional[Participant]:
        for pid, p in self.participants.items():
            if pid != participant_id:
                return p
        return None


class RoomManager:
    def __init__(self):
        self._rooms: dict[str, Room] = {}

    def create_room(self) -> str:
        room_id = secrets.token_urlsafe(6)
        self._rooms[room_id] = Room(id=room_id)
        return room_id

    def get_room(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def remove_participant(self, room_id: str, participant_id: str) -> None:
        room = self._rooms.get(room_id)
        if not room:
            return
        room.participants.pop(participant_id, None)
        if not room.participants:
            self._rooms.pop(room_id, None)

    async def broadcast(
        self,
        room_id: str,
        message: dict,
        exclude: Optional[str] = None,
    ) -> None:
        room = self._rooms.get(room_id)
        if not room:
            return
        for pid, participant in room.participants.items():
            if exclude and pid == exclude:
                continue
            try:
                await participant.websocket.send_json(message)
            except Exception:
                pass

    async def send_to(
        self, room_id: str, participant_id: str, message: dict
    ) -> None:
        room = self._rooms.get(room_id)
        if not room:
            return
        participant = room.participants.get(participant_id)
        if participant:
            try:
                await participant.websocket.send_json(message)
            except Exception:
                pass
