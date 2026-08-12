"use client";

import type { InterpretationMode, SessionSettings } from "@/lib/schemas";
import { DEFAULT_SETTINGS } from "@/lib/schemas";

type Props = {
  settings: SessionSettings;
  onChange: (s: SessionSettings) => void;
  interpretationMode: InterpretationMode;
  onInterpretationModeChange: (m: InterpretationMode) => void;
  sttEnabled: boolean;
  onSttToggle: (v: boolean) => void;
  onClearContext: () => void;
  open: boolean;
  onToggle: () => void;
};

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span>{step < 1 ? value.toFixed(2) : Math.round(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </label>
  );
}

export function SettingsPanel({
  settings,
  onChange,
  interpretationMode,
  onInterpretationModeChange,
  sttEnabled,
  onSttToggle,
  onClearContext,
  open,
  onToggle,
}: Props) {
  const patch = (partial: Partial<SessionSettings>) =>
    onChange({ ...settings, ...partial });

  return (
    <div className="relative">
      <button
        onClick={onToggle}
        className="p-2 rounded-lg bg-surface-light hover:bg-gray-700 text-sm"
        title="Ajustes"
      >
        ⚙ Ajustes
      </button>
      {open && (
        <div className="absolute bottom-full right-0 mb-2 w-80 max-h-[70vh] overflow-y-auto rounded-xl bg-surface border border-gray-600 p-4 space-y-4 shadow-xl z-50">
          <h4 className="font-semibold text-sm">Configuración</h4>

          <Slider
            label="Confianza mínima"
            value={settings.confidence_threshold}
            min={0.1}
            max={1}
            step={0.05}
            onChange={(v) => patch({ confidence_threshold: v })}
          />
          <Slider
            label="Sensibilidad (píxeles)"
            value={settings.motion_pixel_threshold}
            min={100}
            max={5000}
            step={50}
            onChange={(v) => patch({ motion_pixel_threshold: v })}
          />
          <Slider
            label="Corte por silencio (frames)"
            value={settings.still_frames_limit}
            min={5}
            max={40}
            step={1}
            onChange={(v) => patch({ still_frames_limit: v })}
          />
          <Slider
            label="Frames manos (estático)"
            value={settings.static_hands_frames_to_start}
            min={2}
            max={15}
            step={1}
            onChange={(v) => patch({ static_hands_frames_to_start: v })}
          />

          <label className="block text-xs text-gray-400">
            Modo de captura
            <select
              value={settings.capture_mode}
              onChange={(e) =>
                patch({ capture_mode: e.target.value as SessionSettings["capture_mode"] })
              }
              className="mt-1 w-full bg-surface-dark border border-gray-600 rounded px-2 py-1 text-sm"
            >
              <option value="auto">Auto</option>
              <option value="static">Estático</option>
              <option value="dynamic">Dinámico</option>
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={settings.show_landmarks}
              onChange={(e) => patch({ show_landmarks: e.target.checked })}
            />
            Mostrar landmarks
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={sttEnabled}
              onChange={(e) => onSttToggle(e.target.checked)}
            />
            Transcribir mi voz al chat
          </label>

          <div className="space-y-1">
            <p className="text-xs text-gray-400">Modo interpretación (recibir)</p>
            <div className="flex gap-1">
              {(["voice", "text", "both"] as InterpretationMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => onInterpretationModeChange(m)}
                  className={`flex-1 py-1 text-xs rounded ${
                    interpretationMode === m
                      ? "bg-accent text-white"
                      : "bg-surface-dark text-gray-400"
                  }`}
                >
                  {m === "voice" ? "Voz" : m === "text" ? "Texto" : "Ambas"}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={onClearContext}
            className="w-full py-2 text-sm bg-surface-dark border border-gray-600 rounded-lg hover:border-accent"
          >
            Limpiar contexto conversacional
          </button>
        </div>
      )}
    </div>
  );
}

export { DEFAULT_SETTINGS };
