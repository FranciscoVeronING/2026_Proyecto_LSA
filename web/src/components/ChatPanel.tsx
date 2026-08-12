"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/schemas";

type Props = {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  sttInterim?: string;
};

function sourceLabel(source?: string): string {
  switch (source) {
    case "interpretation":
      return "Interpretación";
    case "stt":
      return "Voz";
    default:
      return "Texto";
  }
}

function sourceColor(source?: string): string {
  switch (source) {
    case "interpretation":
      return "border-l-green-500";
    case "stt":
      return "border-l-blue-500";
    default:
      return "border-l-gray-500";
  }
}

export function ChatPanel({ messages, onSend, sttInterim }: Props) {
  const [text, setText] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sttInterim]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <div className="flex flex-col h-full rounded-xl bg-surface overflow-hidden">
      <div className="px-4 py-2 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-300">Chat</h3>
      </div>
      <div className="chat-scroll flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-2 min-h-0">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`pl-3 border-l-2 ${sourceColor(msg.source)} text-sm`}
          >
            <div className="flex gap-2 text-xs text-gray-500 mb-0.5">
              <span>{msg.participant_name || msg.participant_id}</span>
              <span>·</span>
              <span>{sourceLabel(msg.source)}</span>
            </div>
            <p className="text-gray-100">{msg.text}</p>
            {msg.glosses && (
              <p className="text-cyan-400/70 text-xs font-mono mt-0.5">
                {msg.glosses}
              </p>
            )}
          </div>
        ))}
        {sttInterim && (
          <p className="text-gray-500 text-sm italic pl-3">
            {sttInterim}…
          </p>
        )}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={handleSubmit} className="p-3 border-t border-gray-700 flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Escribí un mensaje…"
          className="flex-1 bg-surface-dark border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-accent rounded-lg text-sm font-medium hover:bg-accent/90"
        >
          Enviar
        </button>
      </form>
    </div>
  );
}
