"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const STUN_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

type SignalHandler = (msg: {
  type: "offer" | "answer" | "ice";
  data: RTCSessionDescriptionInit | RTCIceCandidateInit;
}) => void;

export function useWebRTC(
  localStream: MediaStream | null,
  onSignal: (type: "offer" | "answer" | "ice", data: unknown) => void,
  onRemoteStream: (stream: MediaStream) => void,
) {
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);

  const getOrCreatePC = useCallback(() => {
    if (pcRef.current) return pcRef.current;
    const pc = new RTCPeerConnection({ iceServers: STUN_SERVERS });
    pcRef.current = pc;

    pc.ontrack = (ev) => {
      const stream = ev.streams[0];
      setRemoteStream(stream);
      onRemoteStream(stream);
    };

    pc.onicecandidate = (ev) => {
      if (ev.candidate) {
        onSignal("ice", ev.candidate.toJSON());
      }
    };

    return pc;
  }, [onSignal, onRemoteStream]);

  useEffect(() => {
    const pc = getOrCreatePC();
    if (!localStream) return;

    const senders = pc.getSenders();
    localStream.getTracks().forEach((track) => {
      const existing = senders.find((s) => s.track?.kind === track.kind);
      if (existing) {
        existing.replaceTrack(track);
      } else {
        pc.addTrack(track, localStream);
      }
    });
  }, [localStream, getOrCreatePC]);

  const createOffer = useCallback(async () => {
    const pc = getOrCreatePC();
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    onSignal("offer", offer);
  }, [getOrCreatePC, onSignal]);

  const handleSignal = useCallback(
    async (signalType: string, data: unknown) => {
      const pc = getOrCreatePC();
      if (signalType === "offer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data as RTCSessionDescriptionInit));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        onSignal("answer", answer);
      } else if (signalType === "answer") {
        await pc.setRemoteDescription(new RTCSessionDescription(data as RTCSessionDescriptionInit));
      } else if (signalType === "ice") {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(data as RTCIceCandidateInit));
        } catch {
          /* ignore stale candidates */
        }
      }
    },
    [getOrCreatePC, onSignal],
  );

  const close = useCallback(() => {
    pcRef.current?.close();
    pcRef.current = null;
    setRemoteStream(null);
  }, []);

  return { remoteStream, createOffer, handleSignal, close };
}
