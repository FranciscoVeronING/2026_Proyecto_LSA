"""Aplicación FastAPI para la webapp LSA Meet."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

import classifier.config as cfg
from server.models import CreateRoomResponse, RoomStatusResponse
from server.rooms import RoomManager
from server.services.inference_service import SharedInferenceService
from server.services.semantic_service import SharedSemanticService
from server.ws_handler import WebSocketHandler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSES_PATH = cfg.CLASSES_PATH

room_manager = RoomManager()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_idx_to_class: dict = {}
_inference: SharedInferenceService | None = None
_semantic: SharedSemanticService | None = None
_ws_handler: WebSocketHandler | None = None


def _load_classes():
    global _idx_to_class
    if not os.path.exists(_CLASSES_PATH):
        return None, None
    with open(_CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    _idx_to_class = idx_to_class
    return idx_to_class, len(idx_to_class)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference, _semantic, _ws_handler

    idx_to_class, num_classes = _load_classes()
    if idx_to_class is None:
        print("[!] No se encontró mapeo de clases; el clasificador no arrancará.")

    _inference = SharedInferenceService(
        idx_to_class or {}, num_classes or 0, device
    )

    def on_semantic_result(room_id, participant_id, glosses, text):
        if _ws_handler:
            _ws_handler._on_semantic_result(room_id, participant_id, glosses, text)

    _semantic = SharedSemanticService(enabled=True, on_result=on_semantic_result)
    _ws_handler = WebSocketHandler(room_manager, _inference, _semantic, device)
    _inference.set_callback(_ws_handler._on_inference_result)

    print("[*] Servidor LSA Meet listo.")
    yield

    if _inference:
        _inference.stop()
    if _semantic:
        _semantic.stop()


app = FastAPI(title="LSA Meet", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "classifier_ready": _inference.ready if _inference else False,
        "semantic_ready": _semantic.ready if _semantic else False,
        "device": str(device),
    }


@app.post("/api/rooms", response_model=CreateRoomResponse)
async def create_room():
    room_id = room_manager.create_room()
    return CreateRoomResponse(room_id=room_id)


@app.get("/api/rooms/{room_id}", response_model=RoomStatusResponse)
async def room_status(room_id: str):
    room = room_manager.get_room(room_id)
    if not room:
        return RoomStatusResponse(
            room_id=room_id,
            participants=0,
            classifier_ready=_inference.ready if _inference else False,
            semantic_ready=_semantic.ready if _semantic else False,
        )
    return RoomStatusResponse(
        room_id=room_id,
        participants=len(room.participants),
        classifier_ready=_inference.ready if _inference else False,
        semantic_ready=_semantic.ready if _semantic else False,
    )


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    if _ws_handler is None:
        await websocket.close(code=1011)
        return
    await _ws_handler.handle_connection(websocket, room_id)
