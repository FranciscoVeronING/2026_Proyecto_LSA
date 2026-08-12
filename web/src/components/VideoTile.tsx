"use client";

import { useCallback, useEffect, useRef } from "react";

type Props = {
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  stream?: MediaStream | null;
  label: string;
  mirrored?: boolean;
  showLandmarks?: boolean;
  overlayMirrored?: boolean;
  pose?: number[];
  leftHand?: number[];
  rightHand?: number[];
};

function drawConnections(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  connections: [number, number][],
  w: number,
  h: number,
  flipX: boolean,
) {
  ctx.strokeStyle = "#00ff88";
  ctx.lineWidth = 2;
  for (const [a, b] of connections) {
    if (!points[a] || !points[b]) continue;
    const ax = (flipX ? 1 - points[a].x : points[a].x) * w;
    const ay = points[a].y * h;
    const bx = (flipX ? 1 - points[b].x : points[b].x) * w;
    const by = points[b].y * h;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
}

function flatToPoints(flat: number[]): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  for (let i = 0; i < flat.length; i += 3) {
    pts.push({ x: flat[i], y: flat[i + 1] });
  }
  return pts;
}

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
];

export function VideoTile({
  videoRef,
  stream,
  label,
  mirrored = false,
  showLandmarks = false,
  overlayMirrored = false,
  pose = [],
  leftHand = [],
  rightHand = [],
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const internalVideoRef = useRef<HTMLVideoElement>(null);
  const activeVideoRef = videoRef || internalVideoRef;

  useEffect(() => {
    const el = activeVideoRef.current;
    if (el && stream) {
      el.srcObject = stream;
    }
  }, [stream, activeVideoRef]);

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = activeVideoRef.current;
    if (!canvas || !video) return;

    const w = video.clientWidth || 640;
    const h = video.clientHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    if (!showLandmarks) return;

    const flip = overlayMirrored;
    if (pose.length >= 33 * 3) {
      drawConnections(ctx, flatToPoints(pose), POSE_CONNECTIONS, w, h, flip);
    }
    if (leftHand.length >= 21 * 3) {
      drawConnections(ctx, flatToPoints(leftHand), HAND_CONNECTIONS, w, h, flip);
    }
    if (rightHand.length >= 21 * 3) {
      drawConnections(ctx, flatToPoints(rightHand), HAND_CONNECTIONS, w, h, flip);
    }
  }, [showLandmarks, overlayMirrored, pose, leftHand, rightHand, activeVideoRef]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  return (
    <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-gray-700">
      <video
        ref={activeVideoRef as React.RefObject<HTMLVideoElement>}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
        style={{ transform: mirrored ? "scaleX(-1)" : undefined }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      <span className="absolute bottom-2 left-2 text-xs bg-black/60 px-2 py-0.5 rounded">
        {label}
      </span>
    </div>
  );
}
