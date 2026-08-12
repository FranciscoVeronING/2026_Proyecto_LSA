"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  acquireHolistic,
  type HolisticInstance,
  type HolisticResults,
} from "@/lib/mediapipe-loader";

type LandmarkCallback = (data: {
  pose: number[];
  left_hand: number[];
  right_hand: number[];
}) => void;

export function useMediaPipeHolistic(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  leftHanded: boolean,
  enabled: boolean,
  onResults: LandmarkCallback,
) {
  const holisticRef = useRef<HolisticInstance | null>(null);
  const mirrorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const cbRef = useRef(onResults);
  cbRef.current = onResults;

  useEffect(() => {
    if (!enabled) {
      setReady(false);
      return;
    }

    let active = true;

    const onMpResults = (results: HolisticResults) => {
      const toFlat = (lms?: { x: number; y: number; z: number }[]) =>
        lms ? lms.flatMap((lm) => [lm.x, lm.y, lm.z]) : [];

      cbRef.current({
        pose: toFlat(results.poseLandmarks),
        left_hand: toFlat(results.leftHandLandmarks),
        right_hand: toFlat(results.rightHandLandmarks),
      });
    };

    acquireHolistic(onMpResults)
      .then((holistic) => {
        if (!active) return;
        holisticRef.current = holistic;
        setReady(true);
      })
      .catch((err) => {
        console.error("[MediaPipe] Error al cargar Holistic:", err);
        if (active) setReady(false);
      });

    // No llamar close(): Emscripten falla si se reinicializa en la misma sesión.
    return () => {
      active = false;
      setReady(false);
    };
  }, [enabled]);

  const processFrame = useCallback(async () => {
    const video = videoRef.current;
    const holistic = holisticRef.current;
    if (!video || !holistic || video.readyState < 2) return;

    let input: HTMLVideoElement | HTMLCanvasElement = video;

    if (leftHanded) {
      if (!mirrorCanvasRef.current) {
        mirrorCanvasRef.current = document.createElement("canvas");
      }
      const canvas = mirrorCanvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
        input = canvas;
      }
    }

    try {
      await holistic.send({ image: input });
    } catch (err) {
      console.error("[MediaPipe] Error en send:", err);
    }
  }, [videoRef, leftHanded]);

  return { ready, processFrame, mirrorCanvasRef };
}
