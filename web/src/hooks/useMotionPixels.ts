"use client";

import { useCallback, useEffect, useRef } from "react";

const MOTION_W = 320;
const MOTION_H = 240;
const THRESHOLD = 25;

/** Replica el conteo de píxeles en movimiento de OpenCV (main.py). */
export function useMotionPixels(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const prevDataRef = useRef<Uint8ClampedArray | null>(null);
  const countRef = useRef(0);

  const compute = useCallback((): number => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return 0;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return 0;

    canvas.width = MOTION_W;
    canvas.height = MOTION_H;
    ctx.filter = "blur(3.5px)";
    ctx.drawImage(video, 0, 0, MOTION_W, MOTION_H);
    ctx.filter = "none";

    const imgData = ctx.getImageData(0, 0, MOTION_W, MOTION_H);
    const gray = new Uint8ClampedArray(MOTION_W * MOTION_H);

    for (let i = 0; i < gray.length; i++) {
      const o = i * 4;
      gray[i] = Math.round(
        0.299 * imgData.data[o] + 0.587 * imgData.data[o + 1] + 0.114 * imgData.data[o + 2],
      );
    }

    let motionCount = 0;
    if (prevDataRef.current && prevDataRef.current.length === gray.length) {
      for (let i = 0; i < gray.length; i++) {
        if (Math.abs(gray[i] - prevDataRef.current[i]) > THRESHOLD) {
          motionCount++;
        }
      }
      // Escalar a resolución 640×480 (factor 4 en área)
      motionCount *= 4;
    }
    prevDataRef.current = gray;
    countRef.current = motionCount;
    return motionCount;
  }, [videoRef]);

  return { compute, getCount: () => countRef.current };
}
