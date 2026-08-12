"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ServerMessageSchema } from "@/lib/schemas";

type MessageHandler = (type: string, payload: Record<string, unknown>) => void;

export function useWebSocket(roomId: string, onMessage: MessageHandler) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [participantId, setParticipantId] = useState<string | null>(null);
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    if (!roomId) return;

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = process.env.NEXT_PUBLIC_API_HOST || "localhost:8000";
    const url = process.env.NEXT_PUBLIC_WS_URL || `${proto}//${host}/ws/${roomId}`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
      try {
        const raw = JSON.parse(ev.data);
        const parsed = ServerMessageSchema.safeParse(raw);
        if (!parsed.success) return;
        const { type, payload } = parsed.data;
        if (type === "connected" && payload.participant_id) {
          setParticipantId(payload.participant_id as string);
        }
        handlerRef.current(type, payload as Record<string, unknown>);
      } catch {
        /* ignore malformed */
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [roomId]);

  const send = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { connected, participantId, send };
}
