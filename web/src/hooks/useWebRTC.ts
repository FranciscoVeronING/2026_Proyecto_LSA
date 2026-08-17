"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const STUN_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

function attachLocalTracks(pc: RTCPeerConnection, stream: MediaStream | null) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    const sender = pc.getSenders().find((s) => s.track?.kind === track.kind);
    if (sender) {
      void sender.replaceTrack(track);
    } else {
      pc.addTrack(track, stream);
    }
  }
}

export function useWebRTC(
  localStream: MediaStream | null,
  onSignal: (type: "offer" | "answer" | "ice", data: unknown) => void,
  onRemoteStream?: (stream: MediaStream) => void,
) {
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const pendingIceRef = useRef<RTCIceCandidateInit[]>([]);
  const onSignalRef = useRef(onSignal);
  const onRemoteStreamRef = useRef(onRemoteStream);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [remoteAudioTracks, setRemoteAudioTracks] = useState(0);

  localStreamRef.current = localStream;
  onSignalRef.current = onSignal;
  onRemoteStreamRef.current = onRemoteStream;

  const publishRemoteStream = useCallback(() => {
    const stream = remoteStreamRef.current;
    if (!stream) return;
    // Clonar referencia para que React detecte cuando llega audio después de video.
    const snapshot = new MediaStream(stream.getTracks());
    setRemoteAudioTracks(snapshot.getAudioTracks().length);
    setRemoteStream(snapshot);
    onRemoteStreamRef.current?.(snapshot);
  }, []);

  const clearRemoteStream = useCallback(() => {
    remoteStreamRef.current = null;
    setRemoteStream(null);
    setRemoteAudioTracks(0);
  }, []);

  const flushIce = useCallback(async (pc: RTCPeerConnection) => {
    if (!pc.remoteDescription) return;
    const queued = pendingIceRef.current.splice(0);
    for (const candidate of queued) {
      try {
        await pc.addIceCandidate(new RTCIceCandidate(candidate));
      } catch {
        /* ignore stale candidates */
      }
    }
  }, []);

  const getOrCreatePC = useCallback(() => {
    if (pcRef.current) return pcRef.current;
    const pc = new RTCPeerConnection({ iceServers: STUN_SERVERS });
    pcRef.current = pc;

    pc.ontrack = (ev) => {
      if (!remoteStreamRef.current) {
        remoteStreamRef.current = new MediaStream();
      }
      const stream = remoteStreamRef.current;
      if (!stream.getTracks().some((t) => t.id === ev.track.id)) {
        stream.addTrack(ev.track);
      }
      if (ev.track.kind === "audio") {
        ev.track.enabled = true;
      }
      publishRemoteStream();
    };

    pc.onicecandidate = (ev) => {
      if (ev.candidate) {
        onSignalRef.current("ice", ev.candidate.toJSON());
      }
    };

    return pc;
  }, [publishRemoteStream]);

  useEffect(() => {
    const pc = getOrCreatePC();
    const stream = localStreamRef.current;
    if (!stream) return;

    const tracksBefore = pc.getSenders().filter((s) => s.track).length;
    attachLocalTracks(pc, stream);
    const tracksAfter = pc.getSenders().filter((s) => s.track).length;

    if (
      tracksAfter > tracksBefore &&
      pc.signalingState === "stable" &&
      pc.remoteDescription
    ) {
      void (async () => {
        try {
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          onSignalRef.current("offer", offer);
        } catch {
          /* ignore */
        }
      })();
    }
  }, [localStream, getOrCreatePC]);

  const createOffer = useCallback(async () => {
    const pc = getOrCreatePC();
    attachLocalTracks(pc, localStreamRef.current);
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    onSignalRef.current("offer", offer);
  }, [getOrCreatePC]);

  const handleSignal = useCallback(
    async (signalType: string, data: unknown) => {
      const pc = getOrCreatePC();
      if (signalType === "offer") {
        await pc.setRemoteDescription(
          new RTCSessionDescription(data as RTCSessionDescriptionInit),
        );
        await flushIce(pc);
        attachLocalTracks(pc, localStreamRef.current);
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        onSignalRef.current("answer", answer);
      } else if (signalType === "answer") {
        await pc.setRemoteDescription(
          new RTCSessionDescription(data as RTCSessionDescriptionInit),
        );
        await flushIce(pc);
      } else if (signalType === "ice") {
        const candidate = data as RTCIceCandidateInit;
        if (!pc.remoteDescription) {
          pendingIceRef.current.push(candidate);
          return;
        }
        try {
          await pc.addIceCandidate(new RTCIceCandidate(candidate));
        } catch {
          /* ignore stale candidates */
        }
      }
    },
    [getOrCreatePC, flushIce],
  );

  const close = useCallback(() => {
    pcRef.current?.close();
    pcRef.current = null;
    remoteStreamRef.current = null;
    pendingIceRef.current = [];
    setRemoteStream(null);
    setRemoteAudioTracks(0);
  }, []);

  return {
    remoteStream,
    remoteAudioTracks,
    createOffer,
    handleSignal,
    close,
    clearRemoteStream,
  };
}
