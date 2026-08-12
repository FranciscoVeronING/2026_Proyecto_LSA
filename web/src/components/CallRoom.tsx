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

export function CallRoom({ roomId, name, isSigner, leftHanded }: Props) {
  const router = useRouter();
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(true);
  const [micOn, setMicOn] = useState(true);
  const [settings, setSettings] = useState<SessionSettings>(DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [interpretationMode, setInterpretationMode] = useState<InterpretationMode>("both");
  const [sttEnabled, setSttEnabled] = useState(!isSigner);
  const [top3, setTop3] = useState<Top3Item[]>([]);
  const [pendingGlosses, setPendingGlosses] = useState("");
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

  const participantIdRef = useRef<string | null>(null);
  const peersRef = useRef<PeerInfo[]>([]);
  const joinedRef = useRef(false);
  const isInitiatorRef = useRef(false);
  const { say, cancel: cancelSpeech, unlock: unlockSpeech } = useSpeechOutput();
  const interpretationModeRef = useRef(interpretationMode);
  const sayRef = useRef(say);
  const lastSpokenRef = useRef("");

  interpretationModeRef.current = interpretationMode;
  sayRef.current = say;

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

  const { createOffer, handleSignal, close: closeRtc } = useWebRTC(
    localStream,
    sendSignal,
    setRemoteStream,
  );

  const handleWsMessage = useCallback(
    (type: string, payload: Record<string, unknown>) => {
      switch (type) {
        case "connected":
          participantIdRef.current = payload.participant_id as string;
          break;

        case "joined":
          setPeers((payload.peers as PeerInfo[]) || []);
          peersRef.current = (payload.peers as PeerInfo[]) || [];
          isInitiatorRef.current = peersRef.current.length === 0;
          if (isInitiatorRef.current && localStream) {
            setTimeout(() => createOffer(), 500);
          }
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
          break;

        case "signal":
          if ((payload.from as string) !== participantIdRef.current) {
            handleSignal(payload.signal_type as string, payload.data);
          }
          break;

        case "top3": {
          const pid = payload.participant_id as string;
          const fromSelf = pid === participantIdRef.current;
          const peerSigner = peersRef.current.find((p) => p.participant_id === pid)?.is_signer;
          if (fromSelf ? isSigner : peerSigner) {
            setTop3((payload.top3 as Top3Item[]) || []);
          }
          break;
        }

        case "gloss_added": {
          const pid = payload.participant_id as string;
          const fromSelf = pid === participantIdRef.current;
          const peerSigner = peersRef.current.find((p) => p.participant_id === pid)?.is_signer;
          if (fromSelf ? isSigner : peerSigner) {
            setPendingGlosses((payload.pending as string) || "");
          }
          break;
        }

        case "utterance_closed": {
          const spanish = payload.spanish as string;
          const pid = payload.participant_id as string;
          setLastSpanish(spanish);
          maybeSpeakInterpretation(spanish, pid);
          break;
        }

        case "chat_message": {
          const parsed = ChatMessageSchema.safeParse(payload);
          if (parsed.success) {
            setChatMessages((prev) => [...prev, parsed.data]);
            if (parsed.data.source === "interpretation" && parsed.data.text) {
              maybeSpeakInterpretation(
                parsed.data.text,
                payload.participant_id as string,
              );
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
          setRemoteStream(null);
          setPeers([]);
          break;
      }
    },
    [createOffer, handleSignal, maybeSpeakInterpretation, localStream],
  );

  const { connected, participantId, send } = useWebSocket(roomId, handleWsMessage);
  wsSendRef.current = send;

  useEffect(() => {
    if (participantId) participantIdRef.current = participantId;
  }, [participantId]);

  useEffect(() => {
    if (!connected || joinedRef.current) return;
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
    navigator.mediaDevices
      .getUserMedia({
        video: { width: 640, height: 480 },
        audio: { echoCancellation: true, noiseSuppression: true },
      })
      .then((stream) => {
        setLocalStream(stream);
        unlockSpeech();
      })
      .catch(console.error);
  }, [unlockSpeech]);

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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <VideoTile
              videoRef={localVideoRef}
              stream={localStream}
              label={`${name} (vos)`}
              mirrored
              showLandmarks={settings.show_landmarks}
              overlayMirrored={localOverlayMirrored}
              pose={localLandmarks.pose}
              leftHand={localLandmarks.left_hand}
              rightHand={localLandmarks.right_hand}
            />
            <VideoTile
              stream={remoteStream}
              label={peer?.name || "Esperando al otro…"}
              showLandmarks={settings.show_landmarks}
              overlayMirrored={peerOverlayMirrored}
              pose={peerLandmarks?.pose}
              leftHand={peerLandmarks?.left_hand}
              rightHand={peerLandmarks?.right_hand}
            />
          </div>

          {isSigner ? (
            <Top3Panel
              top3={top3}
              pendingGlosses={pendingGlosses}
              threshold={settings.confidence_threshold}
            />
          ) : top3.length > 0 ? (
            <Top3Panel
              top3={top3}
              pendingGlosses={pendingGlosses}
              threshold={settings.confidence_threshold}
            />
          ) : null}

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
