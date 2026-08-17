"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { VideoTile } from "@/components/VideoTile";
import { Top3Panel } from "@/components/Top3Panel";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsPanel, DEFAULT_SETTINGS } from "@/components/SettingsPanel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useWebRTC } from "@/hooks/useWebRTC";
import { useMediaPipeHolistic } from "@/hooks/useMediaPipe";
import { useMotionPixels } from "@/hooks/useMotionPixels";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";
import { useSpeechOutput } from "@/hooks/useSpeechOutput";
import type {
  ChatMessage,
  InterpretationMode,
  PeerInfo,
  SessionSettings,
  Top3Item,
} from "@/lib/schemas";
import { ChatMessageSchema } from "@/lib/schemas";

type Props = {
  roomId: string;
  name: string;
  isSigner: boolean;
  leftHanded: boolean;
};

type VideoCaption = {
  text: string;
  sub?: string;
  kind: "interpretation" | "speech" | "pending" | "typed";
};

export function CallRoom({ roomId, name, isSigner, leftHanded }: Props) {
  const router = useRouter();
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteAudioBlocked, setRemoteAudioBlocked] = useState(false);
  const [cameraOn, setCameraOn] = useState(true);
  const [micOn, setMicOn] = useState(true);
  const [settings, setSettings] = useState<SessionSettings>(DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [interpretationMode, setInterpretationMode] = useState<InterpretationMode>("both");
  const [sttEnabled, setSttEnabled] = useState(false);
  const [localTop3, setLocalTop3] = useState<Top3Item[]>([]);
  const [localPending, setLocalPending] = useState("");
  const [peerTop3, setPeerTop3] = useState<Top3Item[]>([]);
  const [peerPending, setPeerPending] = useState("");
  const [localCaption, setLocalCaption] = useState<VideoCaption | null>(null);
  const [peerCaption, setPeerCaption] = useState<VideoCaption | null>(null);
  const [lastSpanish, setLastSpanish] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [peers, setPeers] = useState<PeerInfo[]>([]);
  const [peerLandmarks, setPeerLandmarks] = useState<{
    pose: number[];
    left_hand: number[];
    right_hand: number[];
    left_handed: boolean;
    mirrored: boolean;
  } | null>(null);
  const [localLandmarks, setLocalLandmarks] = useState<{
    pose: number[];
    left_hand: number[];
    right_hand: number[];
  }>({ pose: [], left_hand: [], right_hand: [] });
  const [mediaError, setMediaError] = useState<string | null>(null);

  const participantIdRef = useRef<string | null>(null);
  const peersRef = useRef<PeerInfo[]>([]);
  const joinedRef = useRef(false);
  const pendingOfferRef = useRef(false);
  const localStreamRef = useRef<MediaStream | null>(null);
  const { say, cancel: cancelSpeech, unlock: unlockSpeech } = useSpeechOutput();
  const interpretationModeRef = useRef(interpretationMode);
  const sayRef = useRef(say);
  const lastSpokenRef = useRef("");

  interpretationModeRef.current = interpretationMode;
  sayRef.current = say;

  const setCaptionForParticipant = useCallback(
    (participantId: string, caption: VideoCaption | null) => {
      if (participantId === participantIdRef.current) {
        setLocalCaption(caption);
      } else {
        setPeerCaption(caption);
      }
    },
    [],
  );

  const maybeSpeakInterpretation = useCallback(
    (spanish: string, participantId: string) => {
      const fromPeer = participantId !== participantIdRef.current;
      const mode = interpretationModeRef.current;

      // Por defecto: voz de la interpretación del otro (el señante → el oyente escucha).
      const shouldSpeak =
        fromPeer || (!fromPeer && isSigner && (mode === "voice" || mode === "both"));

      if (!shouldSpeak || (mode !== "voice" && mode !== "both")) return;
      if (!spanish?.trim() || lastSpokenRef.current === spanish) return;

      lastSpokenRef.current = spanish;
      sayRef.current(spanish, mode);
    },
    [isSigner],
  );

  const sendSignal = useCallback(
    (signalType: "offer" | "answer" | "ice", data: unknown) => {
      wsSendRef.current?.({
        type: "signal",
        signal_type: signalType,
        data,
      });
    },
    [],
  );

  const wsSendRef = useRef<(msg: Record<string, unknown>) => void>(() => {});

  const { remoteStream, remoteAudioTracks, createOffer, handleSignal, close: closeRtc, clearRemoteStream } =
    useWebRTC(localStream, sendSignal);
  const createOfferRef = useRef(createOffer);
  createOfferRef.current = createOffer;

  const scheduleOffer = useCallback(() => {
    pendingOfferRef.current = true;
    if (localStreamRef.current?.getAudioTracks().length) {
      pendingOfferRef.current = false;
      window.setTimeout(() => void createOfferRef.current(), 300);
    }
  }, []);

  const handleWsMessage = useCallback(
    (type: string, payload: Record<string, unknown>) => {
      switch (type) {
        case "connected":
          participantIdRef.current = payload.participant_id as string;
          break;

        case "joined":
          setPeers((payload.peers as PeerInfo[]) || []);
          peersRef.current = (payload.peers as PeerInfo[]) || [];
          break;

        case "peer_joined":
          setPeers((prev) => {
            const next = [
              ...prev,
              {
                participant_id: payload.participant_id as string,
                name: payload.name as string,
                is_signer: payload.is_signer as boolean,
                left_handed: payload.left_handed as boolean,
              },
            ];
            peersRef.current = next;
            return next;
          });
          // El que ya estaba en la sala inicia WebRTC cuando entra el otro.
          scheduleOffer();
          break;

        case "signal":
          if ((payload.from as string) !== participantIdRef.current) {
            handleSignal(payload.signal_type as string, payload.data);
          }
          break;

        case "top3": {
          const pid = payload.participant_id as string;
          const items = (payload.top3 as Top3Item[]) || [];
          if (pid === participantIdRef.current) {
            setLocalTop3(items);
          } else {
            setPeerTop3(items);
          }
          break;
        }

        case "gloss_added": {
          const pid = payload.participant_id as string;
          const pending = (payload.pending as string) || "";
          if (pid === participantIdRef.current) {
            setLocalPending(pending);
          } else {
            setPeerPending(pending);
          }
          break;
        }

        case "utterance_closed": {
          const spanish = payload.spanish as string;
          const glosses = (payload.glosses as string) || "";
          const pid = payload.participant_id as string;
          setLastSpanish(spanish);
          setCaptionForParticipant(pid, {
            text: spanish,
            sub: glosses || undefined,
            kind: "interpretation",
          });
          maybeSpeakInterpretation(spanish, pid);
          break;
        }

        case "chat_message": {
          const parsed = ChatMessageSchema.safeParse(payload);
          if (parsed.success) {
            setChatMessages((prev) => [...prev, parsed.data]);
            if (parsed.data.source === "interpretation" && parsed.data.text) {
              maybeSpeakInterpretation(parsed.data.text, parsed.data.participant_id);
            }
          }
          break;
        }

        case "peer_landmarks":
          setPeerLandmarks({
            pose: (payload.pose as number[]) || [],
            left_hand: (payload.left_hand as number[]) || [],
            right_hand: (payload.right_hand as number[]) || [],
            left_handed: payload.left_handed as boolean,
            mirrored: payload.mirrored as boolean,
          });
          break;

        case "peer_left":
          clearRemoteStream();
          setPeers([]);
          setPeerTop3([]);
          setPeerPending("");
          setPeerCaption(null);
          break;
      }
    },
    [handleSignal, maybeSpeakInterpretation, scheduleOffer, clearRemoteStream, setCaptionForParticipant],
  );

  const { connected, participantId, send } = useWebSocket(roomId, handleWsMessage);
  wsSendRef.current = send;

  useEffect(() => {
    if (participantId) participantIdRef.current = participantId;
  }, [participantId]);

  useEffect(() => {
    if (!connected) {
      joinedRef.current = false;
      return;
    }
    if (joinedRef.current) return;
    joinedRef.current = true;
    send({
      type: "join",
      name,
      is_signer: isSigner,
      left_handed: leftHanded,
      landmarks_already_mirrored: leftHanded,
    });
  }, [connected, send, name, isSigner, leftHanded]);

  useEffect(() => {
    const insecureRemote =
      typeof window !== "undefined" &&
      !window.isSecureContext &&
      !/^(localhost|127\.0\.0\.1)$/i.test(window.location.hostname);

    if (!navigator.mediaDevices?.getUserMedia) {
      setMediaError(
        insecureRemote
          ? "Cámara y micrófono bloqueados: abrí la app por HTTPS (Tailscale Serve) o desde localhost — http://100.x.x.x no alcanza."
          : "Cámara/micrófono no disponibles. Usá Chrome actualizado.",
      );
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        video: { width: 640, height: 480 },
        audio: { echoCancellation: true, noiseSuppression: true },
      })
      .then((stream) => {
        setMediaError(null);
        localStreamRef.current = stream;
        setLocalStream(stream);
        unlockSpeech();
      })
      .catch((err) => {
        console.error(err);
        setMediaError(
          err instanceof DOMException && err.name === "NotAllowedError"
            ? "Permiso denegado: permití cámara y micrófono en Chrome."
            : "No se pudo acceder a cámara/micrófono.",
        );
      });
  }, [unlockSpeech]);

  useEffect(() => {
    if (!localStream) return;
    localStreamRef.current = localStream;
    if (pendingOfferRef.current && localStream.getAudioTracks().length > 0) {
      pendingOfferRef.current = false;
      window.setTimeout(() => void createOfferRef.current(), 300);
    }
  }, [localStream]);

  useEffect(() => {
    const el = remoteAudioRef.current;
    if (!el || !remoteStream) return;

    const audioTracks = remoteStream.getAudioTracks();
    if (audioTracks.length === 0) {
      setRemoteAudioBlocked(true);
      return;
    }

    const audioOnly = new MediaStream(audioTracks);
    el.srcObject = audioOnly;
    el.muted = false;
    el.volume = 1;

    const tryPlay = () => {
      void el.play()
        .then(() => setRemoteAudioBlocked(false))
        .catch(() => setRemoteAudioBlocked(true));
    };

    tryPlay();
    audioTracks.forEach((track) => {
      track.onunmute = tryPlay;
    });
  }, [remoteStream, remoteAudioTracks]);

  const unlockRemoteAudio = useCallback(() => {
    const el = remoteAudioRef.current;
    if (!el || !remoteStream) return;
    const audioTracks = remoteStream.getAudioTracks();
    if (audioTracks.length === 0) return;
    el.srcObject = new MediaStream(audioTracks);
    el.muted = false;
    el.volume = 1;
    void el.play()
      .then(() => setRemoteAudioBlocked(false))
      .catch(() => setRemoteAudioBlocked(true));
  }, [remoteStream]);

  const handleLeaveCall = useCallback(() => {
    cancelSpeech();
    localStream?.getTracks().forEach((t) => t.stop());
    closeRtc();
    router.push("/");
  }, [cancelSpeech, localStream, closeRtc, router]);

  useEffect(() => {
    localStream?.getVideoTracks().forEach((t) => {
      t.enabled = cameraOn;
    });
  }, [cameraOn, localStream]);

  useEffect(() => {
    localStream?.getAudioTracks().forEach((t) => {
      t.enabled = micOn;
    });
  }, [micOn, localStream]);

  const landmarksRef = useRef({ pose: [] as number[], left_hand: [] as number[], right_hand: [] as number[] });
  const sendRef = useRef(send);
  sendRef.current = send;

  const onLandmarks = useCallback((data: { pose: number[]; left_hand: number[]; right_hand: number[] }) => {
    landmarksRef.current = data;
    setLocalLandmarks(data);
  }, []);

  const { ready: mpReady, processFrame } = useMediaPipeHolistic(
    localVideoRef,
    leftHanded,
    isSigner && cameraOn,
    onLandmarks,
  );

  const { compute: computeMotion } = useMotionPixels(localVideoRef);

  useEffect(() => {
    if (!isSigner || !cameraOn || !mpReady) return;
    let active = true;

    const tick = async () => {
      if (!active) return;
      await processFrame();
      const motion = computeMotion();
      const lm = landmarksRef.current;
      sendRef.current({
        type: "landmarks",
        pose: lm.pose,
        left_hand: lm.left_hand,
        right_hand: lm.right_hand,
        motion_pixels: motion,
        mirrored: leftHanded,
      });
    };

    const id = setInterval(tick, 50);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [isSigner, cameraOn, mpReady, processFrame, computeMotion, leftHanded]);

  const handleSttFinal = useCallback(
    (text: string) => {
      send({ type: "chat", text, source: "stt" });
    },
    [send],
  );

  const { interim: sttInterim } = useSpeechRecognition(sttEnabled && micOn, handleSttFinal);

  const handleChatSend = useCallback(
    (text: string) => {
      send({ type: "chat", text, source: "typed" });
    },
    [send],
  );

  const handleSettingsChange = useCallback(
    (s: SessionSettings) => {
      setSettings(s);
      send({ type: "settings", settings: s });
    },
    [send],
  );

  const peer = peers[0];
  const peerOverlayMirrored = peer ? !peer.left_handed : false;
  const localOverlayMirrored = !leftHanded;

  return (
    <div className="min-h-screen p-4 flex flex-col gap-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold">LSA Meet</h1>
          <p className="text-xs text-gray-400">
            {connected ? "Conectado" : "Conectando…"} · Sala {roomId}
          </p>
        </div>
        {lastSpanish && (
          <p className="text-sm text-green-300 max-w-md truncate">
            ES: {lastSpanish}
          </p>
        )}
      </header>

      {mediaError && (
        <div className="rounded-lg border border-amber-600/60 bg-amber-950/40 px-4 py-3 text-sm text-amber-200">
          {mediaError}
        </div>
      )}

      {(remoteAudioBlocked || remoteAudioTracks === 0) && remoteStream && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-sky-600/60 bg-sky-950/40 px-4 py-3 text-sm text-sky-100">
          <span>
            {remoteAudioTracks === 0
              ? "Esperando audio del otro participante…"
              : "Activá el audio del otro participante (Chrome lo bloquea hasta que hagas clic)."}
          </span>
          <button
            type="button"
            onClick={unlockRemoteAudio}
            disabled={remoteAudioTracks === 0}
            className="shrink-0 rounded-lg bg-sky-700 px-3 py-1.5 text-sm hover:bg-sky-600 disabled:opacity-40"
          >
            Activar audio
          </button>
        </div>
      )}

      {/* Audio remoto aparte del video (más fiable en Chrome) */}
      <audio ref={remoteAudioRef} autoPlay playsInline className="hidden" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <VideoTile
              videoRef={localVideoRef}
              stream={localStream}
              label={`${name} (vos)`}
              mirrored
              muted
              subtitle={localCaption?.text}
              subtitleSub={localCaption?.sub}
              subtitleKind={localCaption?.kind}
              showLandmarks={settings.show_landmarks}
              overlayMirrored={localOverlayMirrored}
              pose={localLandmarks.pose}
              leftHand={localLandmarks.left_hand}
              rightHand={localLandmarks.right_hand}
            />
            <VideoTile
              stream={remoteStream}
              label={peer?.name || "Esperando al otro…"}
              muted
              subtitle={peerCaption?.text}
              subtitleSub={peerCaption?.sub}
              subtitleKind={peerCaption?.kind}
              showLandmarks={settings.show_landmarks}
              overlayMirrored={peerOverlayMirrored}
              pose={peerLandmarks?.pose}
              leftHand={peerLandmarks?.left_hand}
              rightHand={peerLandmarks?.right_hand}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Top3Panel
              title={`${name} (vos)${isSigner ? " · señante" : ""}`}
              top3={localTop3}
              pendingGlosses={localPending}
              threshold={settings.confidence_threshold}
              emptyHint={
                isSigner ? "Señá frente a la cámara…" : "No sos señante — sin predicciones propias"
              }
            />
            <Top3Panel
              title={
                peer
                  ? `${peer.name}${peer.is_signer ? " · señante" : ""}`
                  : "Otro participante"
              }
              top3={peerTop3}
              pendingGlosses={peerPending}
              threshold={settings.confidence_threshold}
              emptyHint={
                !peer
                  ? "Esperando al otro…"
                  : peer.is_signer
                    ? "Esperando señas del otro…"
                    : "El otro no es señante"
              }
            />
          </div>

          <div className="flex items-center justify-center gap-3">
            <button
              onClick={() => setCameraOn((v) => !v)}
              className={`px-4 py-2 rounded-lg text-sm ${cameraOn ? "bg-gray-700" : "bg-red-600"}`}
            >
              {cameraOn ? "📷 Cámara" : "📷 Apagada"}
            </button>
            <button
              onClick={() => setMicOn((v) => !v)}
              className={`px-4 py-2 rounded-lg text-sm ${micOn ? "bg-gray-700" : "bg-red-600"}`}
            >
              {micOn ? "🎤 Mic" : "🎤 Mudo"}
            </button>
            <SettingsPanel
              settings={settings}
              onChange={handleSettingsChange}
              interpretationMode={interpretationMode}
              onInterpretationModeChange={(m) => {
                setInterpretationMode(m);
                unlockSpeech();
              }}
              sttEnabled={sttEnabled}
              onSttToggle={setSttEnabled}
              onClearContext={() => send({ type: "clear_context" })}
              open={settingsOpen}
              onToggle={() => setSettingsOpen((v) => !v)}
            />
            <button
              onClick={handleLeaveCall}
              className="px-4 py-2 rounded-lg text-sm bg-red-700 hover:bg-red-600 font-medium"
              title="Salir de la llamada"
            >
              Salir
            </button>
          </div>
        </div>

        <div className="flex flex-col min-h-[300px] h-[500px] lg:h-full lg:max-h-[calc(100vh-6rem)]">
          <ChatPanel
            messages={chatMessages}
            onSend={handleChatSend}
            sttInterim={sttInterim}
          />
        </div>
      </div>
    </div>
  );
}
