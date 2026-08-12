"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type Props = {
  roomId: string;
};

export function JoinForm({ roomId }: Props) {
  const router = useRouter();
  const [step, setStep] = useState<1 | 2>(1);
  const [name, setName] = useState("");
  const [isSigner, setIsSigner] = useState<boolean | null>(null);
  const [leftHanded, setLeftHanded] = useState<boolean | null>(null);

  const canProceedStep1 = name.trim().length > 0 && isSigner !== null;
  const canJoin = leftHanded !== null || isSigner === false;

  const handleJoin = () => {
    const params = new URLSearchParams({
      name: name.trim(),
      isSigner: String(isSigner),
      leftHanded: String(isSigner ? leftHanded : false),
    });
    router.push(`/room/${roomId}/call?${params.toString()}`);
  };

  return (
    <div className="max-w-md mx-auto mt-16 p-8 rounded-2xl bg-surface border border-gray-700 space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Unirse a la llamada</h1>
        <p className="text-gray-400 text-sm mt-1">
          Sala: <span className="font-mono text-accent">{roomId}</span>
        </p>
      </div>

      {step === 1 && (
        <div className="space-y-4">
          <label className="block">
            <span className="text-sm text-gray-400">Tu nombre</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 w-full bg-surface-dark border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-accent"
              placeholder="Ej: Juan"
            />
          </label>

          <div>
            <p className="text-sm text-gray-400 mb-2">¿Sos señante?</p>
            <div className="flex gap-2">
              <button
                onClick={() => setIsSigner(true)}
                className={`flex-1 py-2 rounded-lg border ${
                  isSigner === true
                    ? "border-accent bg-accent/20 text-white"
                    : "border-gray-600 text-gray-400"
                }`}
              >
                Sí, seño en LSA
              </button>
              <button
                onClick={() => setIsSigner(false)}
                className={`flex-1 py-2 rounded-lg border ${
                  isSigner === false
                    ? "border-accent bg-accent/20 text-white"
                    : "border-gray-600 text-gray-400"
                }`}
              >
                No, soy oyente
              </button>
            </div>
          </div>

          <button
            disabled={!canProceedStep1}
            onClick={() => {
              if (isSigner === false) {
                handleJoin();
              } else {
                setStep(2);
              }
            }}
            className="w-full py-3 bg-accent rounded-lg font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90"
          >
            {isSigner === false ? "Entrar a la llamada" : "Continuar"}
          </button>
        </div>
      )}

      {step === 2 && isSigner && (
        <div className="space-y-4">
          <p className="text-sm text-gray-400">¿Qué mano hábil sos?</p>
          <div className="flex gap-2">
            <button
              onClick={() => setLeftHanded(false)}
              className={`flex-1 py-3 rounded-lg border ${
                leftHanded === false
                  ? "border-accent bg-accent/20 text-white"
                  : "border-gray-600 text-gray-400"
              }`}
            >
              Diestro
            </button>
            <button
              onClick={() => setLeftHanded(true)}
              className={`flex-1 py-3 rounded-lg border ${
                leftHanded === true
                  ? "border-accent bg-accent/20 text-white"
                  : "border-gray-600 text-gray-400"
              }`}
            >
              Zurdo
            </button>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setStep(1)}
              className="flex-1 py-3 border border-gray-600 rounded-lg text-gray-400"
            >
              Volver
            </button>
            <button
              disabled={!canJoin}
              onClick={handleJoin}
              className="flex-1 py-3 bg-accent rounded-lg font-medium disabled:opacity-40 hover:bg-accent/90"
            >
              Entrar a la llamada
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
