"use client";

import { useCallback, useEffect, useRef } from "react";
import type { InterpretationMode } from "@/lib/schemas";

function pickSpanishVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | undefined {
  return (
    voices.find((v) => v.lang === "es-AR") ||
    voices.find((v) => v.lang.startsWith("es-AR")) ||
    voices.find((v) => v.lang === "es-419") ||
    voices.find((v) => v.lang.startsWith("es-")) ||
    voices.find((v) => v.lang.includes("es"))
  );
}

export function useSpeechOutput() {
  const queueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const voicesRef = useRef<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    const refreshVoices = () => {
      voicesRef.current = speechSynthesis.getVoices();
    };

    refreshVoices();
    speechSynthesis.addEventListener("voiceschanged", refreshVoices);

    // Chrome a veces inicia pausado; requiere gesto del usuario pero esto ayuda después.
    if (speechSynthesis.paused) {
      speechSynthesis.resume();
    }

    return () => {
      speechSynthesis.removeEventListener("voiceschanged", refreshVoices);
    };
  }, []);

  const speakNext = useCallback(() => {
    if (speakingRef.current || queueRef.current.length === 0) return;
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    const text = queueRef.current.shift()!;
    speakingRef.current = true;

    if (speechSynthesis.paused) {
      speechSynthesis.resume();
    }

    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "es-AR";
    utter.rate = 1;

    const voice = pickSpanishVoice(voicesRef.current);
    if (voice) utter.voice = voice;

    utter.onend = () => {
      speakingRef.current = false;
      speakNext();
    };
    utter.onerror = (ev) => {
      console.warn("[TTS] Error:", ev.error);
      speakingRef.current = false;
      speakNext();
    };

    speechSynthesis.speak(utter);
  }, []);

  const say = useCallback(
    (text: string, mode: InterpretationMode) => {
      if (mode === "text" || !text?.trim()) return;
      queueRef.current.push(text.trim());
      speakNext();
    },
    [speakNext],
  );

  const cancel = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      speechSynthesis.cancel();
    }
    queueRef.current = [];
    speakingRef.current = false;
  }, []);

  /** Desbloquea TTS en navegadores que exigen interacción previa (Chrome). */
  const unlock = useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    speechSynthesis.resume();
    const u = new SpeechSynthesisUtterance("");
    u.volume = 0;
    speechSynthesis.speak(u);
    speechSynthesis.cancel();
  }, []);

  return { say, cancel, unlock };
}
