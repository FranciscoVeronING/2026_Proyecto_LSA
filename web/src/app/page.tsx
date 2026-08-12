"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { createRoom } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [joinId, setJoinId] = useState("");
  const [error, setError] = useState("");

  const createMutation = useMutation({
    mutationFn: createRoom,
    onSuccess: (roomId) => router.push(`/room/${roomId}`),
    onError: () => setError("No se pudo crear la sala. ¿Está corriendo el backend?"),
  });

  const handleJoin = () => {
    const id = joinId.trim();
    if (!id) return;
    router.push(`/room/${id}`);
  };

  return (
    <main className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-lg w-full p-8 rounded-2xl bg-surface border border-gray-700 space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-bold">LSA Meet</h1>
          <p className="text-gray-400 mt-2">
            Videollamada con interpretación de Lengua de Señas Argentina en tiempo real
          </p>
        </div>

        <button
          onClick={() => createMutation.mutate()}
          disabled={createMutation.isPending}
          className="w-full py-3 bg-accent rounded-xl font-semibold hover:bg-accent/90 disabled:opacity-50"
        >
          {createMutation.isPending ? "Creando…" : "Crear nueva sala"}
        </button>

        <div className="flex items-center gap-3 text-gray-500 text-sm">
          <div className="flex-1 h-px bg-gray-700" />
          <span>o unirse</span>
          <div className="flex-1 h-px bg-gray-700" />
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={joinId}
            onChange={(e) => setJoinId(e.target.value)}
            placeholder="Código de sala"
            className="flex-1 bg-surface-dark border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-accent"
          />
          <button
            onClick={handleJoin}
            className="px-5 py-2 border border-gray-600 rounded-lg hover:border-accent"
          >
            Unirse
          </button>
        </div>

        {error && <p className="text-red-400 text-sm text-center">{error}</p>}

        <p className="text-xs text-gray-500 text-center">
          Requiere Chrome, cámara y micrófono. Backend en{" "}
          <code className="text-gray-400">python run_server.py</code>
        </p>
      </div>
    </main>
  );
}
