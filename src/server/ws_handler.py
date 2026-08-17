"""Handler WebSocket por sala."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Optional

import torch
from pydantic import ValidationError

from app.pipeline import ParticipantPipeline
from app.settings import SessionSettings
from server.frame_processor import decode_jpeg_b64, extract_raw_landmarks_from_frame
from server.models import (
    ChatPayload,
    FramePayload,
    JoinPayload,
    LandmarksPayload,
    SettingsPayload,
    SignalPayload,
)
from server.rooms import Participant, RoomManager

if TYPE_CHECKING:
    from server.services.inference_service import SharedInferenceService
    from server.services.semantic_service import SharedSemanticService


class WebSocketHandler:
    def __init__(
        self,
        room_manager: RoomManager,
        inference: "SharedInferenceService",
        semantic: "SharedSemanticService",
        device: torch.device,
    ):
        self.room_manager = room_manager
        self.inference = inference
        self.semantic = semantic
        self.device = device
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room_loops: dict[str, asyncio.AbstractEventLoop] = {}

    def _schedule(self, coro) -> None:
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)

    def _on_inference_result(self, room_id: str, participant_id: str, top3: list) -> None:
        async def _broadcast():
            room = self.room_manager.get_room(room_id)
            if not room:
                return
            participant = room.participants.get(participant_id)
            if not participant or not participant.pipeline:
                return
            events = participant.pipeline.apply_inference_result(top3)
            for event in events:
                await self.room_manager.broadcast(
                    room_id,
                    {
                        "type": event.type,
                        "payload": {**event.payload, "participant_id": participant_id},
                    },
                )

        self._schedule(_broadcast())

    def _on_semantic_result(
        self, room_id: str, participant_id: str, glosses: str, text: str
    ) -> None:
        async def _broadcast():
            room = self.room_manager.get_room(room_id)
            if not room:
                return
            participant = room.participants.get(participant_id)
            if participant and participant.pipeline:
                participant.pipeline.apply_semantic_result(text, glosses, busy=False)

            await self.room_manager.broadcast(
                room_id,
                {
                    "type": "utterance_closed",
                    "payload": {
                        "participant_id": participant_id,
                        "participant_name": participant.name if participant else "",
                        "spanish": text,
                        "glosses": glosses,
                    },
                },
            )
            await self.room_manager.broadcast(
                room_id,
                {
                    "type": "chat_message",
                    "payload": {
                        "participant_id": participant_id,
                        "participant_name": participant.name if participant else "",
                        "text": text,
                        "glosses": glosses,
                        "source": "interpretation",
                        "is_signer": True,
                    },
                },
            )

        self._schedule(_broadcast())

    async def handle_connection(self, websocket, room_id: str) -> None:
        self._loop = asyncio.get_running_loop()
        room = self.room_manager.ensure_room(room_id)

        if room.is_full():
            await websocket.close(code=4003, reason="Sala llena")
            return

        participant_id = str(uuid.uuid4())[:8]
        participant: Participant | None = None

        try:
            await websocket.accept()
            print(f"[ws] conectado sala={room_id} participante={participant_id}", flush=True)
            await websocket.send_json(
                {
                    "type": "connected",
                    "payload": {
                        "participant_id": participant_id,
                        "room_id": room_id,
                        "classifier_ready": self.inference.ready,
                        "semantic_ready": self.semantic.ready,
                    },
                }
            )

            async for raw in websocket.iter_text():
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "join":
                    await self._handle_join(
                        websocket, room, room_id, participant_id, data, participant
                    )
                    participant = room.participants.get(participant_id)

                elif msg_type == "landmarks" and participant:
                    await self._handle_landmarks(room_id, participant, data)

                elif msg_type == "frame" and participant:
                    await self._handle_frame(room_id, participant, data)

                elif msg_type == "chat" and participant:
                    await self._handle_chat(room_id, participant, data)

                elif msg_type == "settings" and participant:
                    await self._handle_settings(room_id, participant, data)

                elif msg_type == "signal" and participant:
                    await self._handle_signal(room_id, participant_id, data)

                elif msg_type == "clear_context" and participant:
                    self.semantic.clear_context(room_id)
                    await self.room_manager.broadcast(
                        room_id, {"type": "context_cleared", "payload": {}}
                    )

        finally:
            print(
                f"[ws] desconectado sala={room_id} participante={participant_id}",
                flush=True,
            )
            self.room_manager.remove_participant(room_id, participant_id)
            if participant:
                other = room.other_participant(participant_id)
                if other:
                    await self.room_manager.send_to(
                        room_id,
                        other.id,
                        {
                            "type": "peer_left",
                            "payload": {"participant_id": participant_id},
                        },
                    )

    async def _handle_join(
        self, websocket, room, room_id, participant_id, data, participant
    ):
        try:
            payload = JoinPayload.model_validate(data)
        except ValidationError:
            return

        def on_enqueue(pid, tensor):
            self.inference.enqueue(room_id, pid, tensor)

        def on_utterance_closed(pid, glosses):
            self.semantic.submit(room_id, pid, glosses)

        pipeline = ParticipantPipeline(
            participant_id=participant_id,
            left_handed=payload.left_handed,
            is_signer=payload.is_signer,
            device=self.device,
            on_enqueue=on_enqueue,
            on_utterance_closed=on_utterance_closed,
            landmarks_already_mirrored=payload.landmarks_already_mirrored,
        )

        participant = Participant(
            id=participant_id,
            name=payload.name,
            is_signer=payload.is_signer,
            left_handed=payload.left_handed,
            websocket=websocket,
            pipeline=pipeline,
            landmarks_already_mirrored=payload.landmarks_already_mirrored,
        )
        room.participants[participant_id] = participant

        await self.room_manager.broadcast(
            room_id,
            {
                "type": "peer_joined",
                "payload": {
                    "participant_id": participant_id,
                    "name": payload.name,
                    "is_signer": payload.is_signer,
                    "left_handed": payload.left_handed,
                },
            },
            exclude=participant_id,
        )

        peers = []
        for pid, p in room.participants.items():
            if pid != participant_id:
                peers.append(
                    {
                        "participant_id": pid,
                        "name": p.name,
                        "is_signer": p.is_signer,
                        "left_handed": p.left_handed,
                    }
                )

        await websocket.send_json(
            {
                "type": "joined",
                "payload": {
                    "participant_id": participant_id,
                    "peers": peers,
                },
            }
        )

    async def _handle_landmarks(self, room_id, participant, data):
        try:
            payload = LandmarksPayload.model_validate(data)
        except ValidationError:
            return

        if not participant.pipeline:
            return

        events = participant.pipeline.process_landmarks(
            pose=payload.pose,
            left_hand=payload.left_hand,
            right_hand=payload.right_hand,
            motion_pixels=payload.motion_pixels,
        )

        for event in events:
            await self.room_manager.broadcast(
                room_id,
                {
                    "type": event.type,
                    "payload": {**event.payload, "participant_id": participant.id},
                },
            )

        await self.room_manager.broadcast(
            room_id,
            {
                "type": "peer_landmarks",
                "payload": {
                    "participant_id": participant.id,
                    "left_handed": participant.left_handed,
                    "mirrored": payload.mirrored,
                    "pose": payload.pose,
                    "left_hand": payload.left_hand,
                    "right_hand": payload.right_hand,
                },
            },
            exclude=participant.id,
        )

    async def _handle_frame(self, room_id, participant, data):
        try:
            payload = FramePayload.model_validate(data)
        except ValidationError:
            return

        frame = decode_jpeg_b64(payload.image_b64)
        if frame is None:
            return

        pose, left_hand, right_hand = extract_raw_landmarks_from_frame(
            frame, mirrored=payload.mirrored
        )

        if not participant.pipeline:
            return

        events = participant.pipeline.process_landmarks(
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            motion_pixels=payload.motion_pixels,
        )

        for event in events:
            await self.room_manager.broadcast(
                room_id,
                {
                    "type": event.type,
                    "payload": {**event.payload, "participant_id": participant.id},
                },
            )

    async def _handle_chat(self, room_id, participant, data):
        try:
            payload = ChatPayload.model_validate(data)
        except ValidationError:
            return

        if payload.source == "stt":
            self.semantic.add_hearing(room_id, payload.text)

        await self.room_manager.broadcast(
            room_id,
            {
                "type": "chat_message",
                "payload": {
                    "participant_id": participant.id,
                    "participant_name": participant.name,
                    "text": payload.text,
                    "source": payload.source,
                    "is_signer": participant.is_signer,
                },
            },
        )

    async def _handle_settings(self, room_id, participant, data):
        try:
            payload = SettingsPayload.model_validate(data)
        except ValidationError:
            return

        if participant.pipeline:
            settings = SessionSettings.from_dict(payload.settings)
            participant.pipeline.update_settings(settings)
            await websocket_send_settings_ack(participant, settings)

    async def _handle_signal(self, room_id, participant_id, data):
        try:
            payload = SignalPayload.model_validate(data)
        except ValidationError:
            return

        room = self.room_manager.get_room(room_id)
        if not room:
            return
        other = room.other_participant(participant_id)
        if other:
            await self.room_manager.send_to(
                room_id,
                other.id,
                {
                    "type": "signal",
                    "payload": {
                        "from": participant_id,
                        "signal_type": payload.signal_type,
                        "data": payload.data,
                    },
                },
            )


async def websocket_send_settings_ack(participant, settings):
    try:
        await participant.websocket.send_json(
            {"type": "settings_ack", "payload": settings.to_dict()}
        )
    except Exception:
        pass
