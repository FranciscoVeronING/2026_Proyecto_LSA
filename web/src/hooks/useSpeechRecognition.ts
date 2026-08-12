"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useSpeechRecognition(
  enabled: boolean,
  onFinal: (text: string) => void,
  onInterim?: (text: string) => void,
) {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    if (!enabled) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    const SpeechRecognitionCtor =
      (window as unknown as { SpeechRecognition?: typeof SpeechRecognition }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: typeof SpeechRecognition }).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      console.warn("[STT] SpeechRecognition no disponible (usar Chrome)");
      return;
    }

    const rec = new SpeechRecognitionCtor();
    rec.lang = "es-AR";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      let interimText = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const result = ev.results[i];
        if (result.isFinal) {
          const text = result[0].transcript.trim();
          if (text) onFinalRef.current(text);
          setInterim("");
        } else {
          interimText += result[0].transcript;
        }
      }
      if (interimText) {
        setInterim(interimText);
        onInterim?.(interimText);
      }
    };

    rec.onend = () => {
      if (enabled) {
        try {
          rec.start();
        } catch {
          /* already started */
        }
      }
    };

    rec.onerror = () => setListening(false);

    try {
      rec.start();
      setListening(true);
    } catch {
      /* mic permission pending */
    }

    recognitionRef.current = rec;

    return () => {
      rec.stop();
      recognitionRef.current = null;
      setListening(false);
    };
  }, [enabled, onInterim]);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    setListening(false);
  }, []);

  return { listening, interim, stop };
}
