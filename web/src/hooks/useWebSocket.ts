"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { backendHost } from "@/lib/api";
import { ServerMessageSchema } from "@/lib/schemas";

type MessageHandler = (type: string, payload: Record<string, unknown>) => void;

const RECONNECT_MS = 1500;

export function useWebSocket(roomId: string, onMessage: MessageHandler) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [participantId, setParticipantId] = useState<string | null>(null);
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    if (!roomId) return;

    intentionalCloseRef.current = false;
    let disposed = false;

    const clearReconnect = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const connect = () => {
      if (disposed) return;
      clearReconnect();

      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url =
        process.env.NEXT_PUBLIC_WS_URL || `${proto}//${backendHost()}/ws/${roomId}`;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        setConnected(true);
      };

      ws.onclose = () => {
        setConnected(false);
        setParticipantId(null);
        if (wsRef.current === ws) wsRef.current = null;

        if (!disposed && !intentionalCloseRef.current) {
          reconnectTimerRef.current = window.setTimeout(connect, RECONNECT_MS);
        }
      };

      ws.onerror = () => {
        setConnected(false);
      };

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
    };

    connect();

    return () => {
      disposed = true;
      intentionalCloseRef.current = true;
      clearReconnect();
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "unmount");
      }
    };
  }, [roomId]);

  const send = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { connected, participantId, send };
}
