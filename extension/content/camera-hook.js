/**
 * getUserMedia hook in the MAIN world.
 * With LSA off, leave the native stream (otherwise Meet says the camera is blocked).
 * With LSA on, Meet receives a canvas (captureStream).
 * Hearing mode: left half camera, right half skeleton, same height.
 */
(function () {
  if (window.__lsaGumHooked) return;
  window.__lsaGumHooked = true;
  try {
    const leftover = document.getElementById("lsa-oyente-subs");
    if (leftover) leftover.remove();
  } catch (_) {}

  const origGetUserMedia = MediaDevices.prototype.getUserMedia;
  const OrigPC = window.RTCPeerConnection;
  const peerConnections = new Set();
  if (OrigPC) {
    window.RTCPeerConnection = class extends OrigPC {
      constructor(...args) {
        super(...args);
        peerConnections.add(this);
        this.addEventListener("connectionstatechange", () => {
          const st = this.connectionState;
          if (st === "closed" || st === "failed") peerConnections.delete(this);
        });
      }
    };
  }
  const SUBTITLE_HOLD_MS = 8000;
  const HEARING_WORKLET = `
class LsaHearingProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) {
      const copy = new Float32Array(ch.length);
      copy.set(ch);
      this.port.postMessage(copy, [copy.buffer]);
    }
    return true;
  }
}
registerProcessor("lsa-hearing", LsaHearingProcessor);
`;
  const POSE_EDGES = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
    [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [15, 17], [16, 18],
  ];
  const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17],
  ];
  let enabled = false;
  let hearingMode = false;
  let hearingInput = "both";
  let showSubtitles = true;
  let showLandmarks = false;
  let landmarks = null;
  let caption = { spanish: "", glosses: "" };
  let avatarClip = null;
  const avatarQueue = [];
  const AVATAR_QUEUE_MAX = 20;
  let lastCaptionLine = "";
  let pipeline = null;
  let extraAudio = null;
  let callMix = null;

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "lsa-ext") return;
    if (data.type === "LSA_SET_ENABLED") {
      const want = Boolean(data.enabled);
      hearingMode = Boolean(want && data.hearing);
      if (!hearingMode) {
        stopHearingTap();
        avatarClip = null;
        avatarQueue.length = 0;
        lastCaptionLine = "";
      }
      if (want && !enabled) {
        enabled = true;
        wrapExistingMeetCamera();
        if (hearingMode) {
          syncHearingCapture();
          [300, 1000, 2500].forEach((ms) => {
            setTimeout(() => {
              if (enabled && hearingMode) {
                wrapExistingMeetCamera();
                syncHearingCapture();
              }
            }, ms);
          });
        }
        syncCanvasToMeet();
      } else if (!want && enabled) {
        enabled = false;
        stopCallVoice();
        document.querySelectorAll("video").forEach((v) => clearPreviewFix(v));
        restoreNativeCamera();
      } else {
        enabled = want;
        if (enabled) {
          wrapExistingMeetCamera();
          if (hearingMode) syncHearingCapture();
          syncCanvasToMeet();
        }
      }
    }
    if (data.type === "LSA_PREFS") {
      if (typeof data.showSubtitles === "boolean") showSubtitles = data.showSubtitles;
      if (data.hearingInput === "voice" || data.hearingInput === "text" || data.hearingInput === "both") {
        hearingInput = data.hearingInput;
        syncHearingCapture();
      }
      if (typeof data.showLandmarks === "boolean") {
        showLandmarks = data.showLandmarks;
        if (!showLandmarks) landmarks = null;
      }
    }
    if (data.type === "LSA_LANDMARKS") {
      landmarks = data.landmarks || null;
    }
    if (data.type === "LSA_SPEECH_STOP") stopHearingTap();
    if (data.type === "LSA_MIC_QUERY") {
      window.postMessage(
        { source: "lsa-page", type: "LSA_MIC_STATE", id: data.id, open: callMicOpen() },
        "*"
      );
    }
    if (data.type === "LSA_TTS_STOP") stopCallVoice();
    if (data.type === "LSA_TTS_WAV" && data.wavB64) playCallVoice(data.wavB64, data.id);
    if (data.type === "LSA_AVATAR" && data.clip && data.clip.ok && window.LsaAvatar) {
      enqueueAvatarClip(data.clip, lastCaptionLine || caption.spanish);
    }
    if (data.type === "LSA_CAPTION") {
      // Only `line`: hearing sends semantic glosses; signer sends Spanish.
      // Never use raw classifier `glosses`.
      const line = typeof data.line === "string" ? data.line.trim() : "";
      if (line) {
        lastCaptionLine = line;
        if (!avatarBusy()) {
          caption.spanish = line;
          if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
          window.__lsaCaptionTimer = setTimeout(() => {
            if (!avatarBusy() && !avatarQueue.length) caption.spanish = "";
          }, SUBTITLE_HOLD_MS);
        }
      }
    }
  });

  let hearingTap = null;

  function liveAudioTracks() {
    const tracks = [];
    const seen = new Set();
    const add = (stream) => {
      if (!stream) return;
      stream.getAudioTracks().forEach((t) => {
        if (t.readyState !== "live" || seen.has(t.id)) return;
        seen.add(t.id);
        tracks.push(t);
      });
    };
    if (pipeline) add(pipeline.real);
    add(extraAudio);
    return tracks;
  }

  /**
   * Mic de Meet mezclado con la voz en off. La locución también sale por
   * los parlantes de esta PC. El micrófono no, para no hacer eco.
   * @param {MediaStream|null} micStream
   * @returns {MediaStream|null}
   */
  function ensureCallMix(micStream) {
    if (!callMix || callMix.ctx.state === "closed") {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const dest = ctx.createMediaStreamDestination();
      const ttsGain = ctx.createGain();
      ttsGain.gain.value = 1;
      ttsGain.connect(dest);
      ttsGain.connect(ctx.destination);
      callMix = { ctx, dest, ttsGain, source: null, mics: new Set() };
      const kick = () => ctx.resume().catch(() => {});
      if (ctx.state === "suspended") {
        kick();
        window.addEventListener("pointerdown", kick, { once: true, capture: true });
      }
      ctx.onstatechange = () => {
        if (ctx.state === "running") attachMixToPipeline();
      };
    }
    if (micStream) {
      micStream.getAudioTracks().forEach((track) => {
        if (!track || callMix.mics.has(track.id)) return;
        callMix.mics.add(track.id);
        try {
          const src = callMix.ctx.createMediaStreamSource(new MediaStream([track]));
          src.connect(callMix.dest);
        } catch (_) {}
      });
    }
    return callMix.dest.stream;
  }

  function attachMixToPipeline() {
    if (hearingMode || !pipeline || !pipeline.out || !callMix || callMix.ctx.state !== "running") return;
    const mixedTrack = callMix.dest.stream.getAudioTracks()[0];
    if (!mixedTrack) return;
    const current = pipeline.out.getAudioTracks();
    if (!current.some((t) => t.id === mixedTrack.id)) {
      current.forEach((t) => {
        try {
          pipeline.out.removeTrack(t);
        } catch (_) {}
      });
      try {
        pipeline.out.addTrack(mixedTrack);
      } catch (_) {}
    }
    syncCanvasToMeet();
  }

  function stopCallVoice() {
    if (!callMix || !callMix.source) return;
    try {
      callMix.source.stop();
    } catch (_) {}
    callMix.source = null;
  }

  /** true si Meet está mandando el micrófono a la llamada. */
  function callMicOpen() {
    if (meetUiMicMuted() === true) return false;
    const rtc = outboundKindLive("audio");
    if (rtc === false) return false;
    const tracks = liveAudioTracks();
    if (tracks.some((t) => t.muted === true)) return false;
    if (tracks.length && tracks.every((t) => t.enabled === false)) return false;
    if (meetUiMicMuted() === false) return true;
    if (rtc === true) return true;
    return tracks.some((t) => t.enabled !== false && t.readyState === "live");
  }

  function wavBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i) & 0xff;
    return bytes.buffer;
  }

  function playCallVoice(b64, id) {
    if (!b64 || !callMicOpen()) {
      window.postMessage({ source: "lsa-page", type: "LSA_VOICE_BLOCKED" }, "*");
      return;
    }
    if (!callMix) {
      window.postMessage(
        { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: "la llamada todavía no tiene audio" },
        "*"
      );
      return;
    }
    const mix = callMix;
    const start = () => {
      attachMixToPipeline();
      mix.ctx.decodeAudioData(
        wavBytes(b64),
        (buffer) => {
          if (!callMicOpen()) {
            window.postMessage({ source: "lsa-page", type: "LSA_VOICE_BLOCKED" }, "*");
            return;
          }
          stopCallVoice();
          const source = mix.ctx.createBufferSource();
          source.buffer = buffer;
          source.connect(mix.ttsGain);
          mix.source = source;
          source.onended = () => {
            if (mix.source === source) mix.source = null;
          };
          try {
            source.start();
            window.postMessage({ source: "lsa-page", type: "LSA_TTS_STARTED", id }, "*");
          } catch (_) {
            window.postMessage(
              { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: "no se pudo reproducir la voz en off" },
              "*"
            );
          }
        },
        () => {
          window.postMessage(
            { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: "el audio de la voz en off no se pudo leer" },
            "*"
          );
        }
      );
    };
    if (mix.ctx.state === "suspended") mix.ctx.resume().then(start).catch(start);
    else start();
  }

  /**
   * true = Meet's button says the microphone is muted.
   * false = the button shows it on. null = the control was not found.
   * “Turn off / mute” is the action: the mic is on.
   * “Turn on the microphone” also counts (there is an extra word in the middle).
   */
  function meetUiMicMuted() {
    const els = document.querySelectorAll("button, [role='button']");
    let fallback = null;
    for (let i = 0; i < els.length; i++) {
      const el = els[i];
      const box = el.getBoundingClientRect();
      if (box.width < 2 || box.height < 2) continue;
      const lab = (
        (el.getAttribute("aria-label") || "") +
        " " +
        (el.getAttribute("data-tooltip") || "") +
        " " +
        (el.getAttribute("title") || "") +
        " " +
        (el.innerText || "").slice(0, 80)
      )
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "");
      const looksMic = /microfono|microphone|\bmic\b|mic_off|mic_none/.test(lab);
      if (!looksMic) continue;
      const flagged = el.getAttribute("data-is-muted");
      if (flagged === "true") return true;
      if (flagged === "false") return false;
      const willUnmute = /\bactivar\b|\bturn on\b|\bunmute\b|\bsilenciado\b|\bmuted\b|mic_off/.test(lab);
      const willMute = /\bdesactivar\b|\bturn off\b|\bsilenciar\b|\bmute\b/.test(lab);
      if (willMute && !willUnmute) {
        fallback = false;
        continue;
      }
      if (willUnmute && !/\bdesactivar\b|\bturn off\b|\bsilenciar\b/.test(lab)) return true;
      if (willUnmute) fallback = true;
    }
    return fallback;
  }

  /** true = camera off in Meet's UI. */
  function meetUiCamOff() {
    const els = document.querySelectorAll("button, [role='button']");
    for (let i = 0; i < els.length; i++) {
      const el = els[i];
      const lab = (
        (el.getAttribute("aria-label") || "") +
        " " +
        (el.getAttribute("data-tooltip") || "") +
        " " +
        (el.getAttribute("title") || "")
      ).toLowerCase();
      if (!/cámara|camara|camera/.test(lab)) continue;
      if (/micrófono|microfono|microphone|\bmic\b/.test(lab)) continue;
      const flagged = el.getAttribute("data-is-muted");
      if (flagged === "true") return true;
      if (flagged === "false") return false;
      if (/desactivar cám|desactivar cam|turn off cam/.test(lab)) return false;
      if (/activar cám|activar cam|turn on cam/.test(lab)) return true;
    }
    return null;
  }

  function senderKindActive(sender, kind) {
    if (!sender) return false;
    try {
      const enc = sender.getParameters().encodings;
      if (enc && enc.length && enc.every((e) => e.active === false)) return false;
    } catch (_) {}
    const t = sender.track;
    return Boolean(t && t.kind === kind && t.readyState === "live" && t.enabled !== false);
  }

  /**
   * true = hay media de `kind` saliendo a la llamada; false = mute WebRTC;
   * null = there are no senders yet (before joining).
   * @param {"audio"|"video"} kind
   */
  function outboundKindLive(kind) {
    let saw = false;
    for (const pc of peerConnections) {
      let transceivers = [];
      try {
        transceivers = pc.getTransceivers() || [];
      } catch (_) {
        transceivers = [];
      }
      for (let i = 0; i < transceivers.length; i++) {
        const tr = transceivers[i];
        const recv = tr.receiver && tr.receiver.track;
        const send = tr.sender && tr.sender.track;
        const match =
          (send && send.kind === kind) ||
          (recv && recv.kind === kind) ||
          tr.kind === kind;
        if (!match) continue;
        saw = true;
        if (senderKindActive(tr.sender, kind)) return true;
      }
      if (transceivers.length) continue;
      let senders = [];
      try {
        senders = pc.getSenders() || [];
      } catch (_) {
        senders = [];
      }
      for (let j = 0; j < senders.length; j++) {
        const sender = senders[j];
        if (!(sender.track && sender.track.kind === kind)) continue;
        saw = true;
        if (senderKindActive(sender, kind)) return true;
      }
    }
    if (saw) return false;
    return null;
  }

  function liveVideoTracks() {
    const tracks = [];
    if (!pipeline || !pipeline.real) return tracks;
    pipeline.real.getVideoTracks().forEach((t) => {
      if (t.readyState === "live") tracks.push(t);
    });
    return tracks;
  }

  function meetCamOpen() {
    if (meetUiCamOff() === true) return false;
    const rtc = outboundKindLive("video");
    if (rtc === false) return false;
    const tracks = liveVideoTracks();
    if (tracks.length && tracks.every((t) => t.enabled === false)) return false;
    if (meetUiCamOff() === false) return true;
    if (rtc === true) return true;
    return tracks.some((t) => t.enabled !== false);
  }

  let lastPostedCam = null;
  function publishCamOpen() {
    const open = meetCamOpen();
    if (open !== lastPostedCam) {
      lastPostedCam = open;
      window.postMessage({ source: "lsa-page", type: "LSA_CAMERA", open }, "*");
    }
    return open;
  }

  function syncHearingCapture() {
    if (hearingMode && hearingInput !== "text") startHearingTap();
    else stopHearingTap();
  }

  let hearingTapGen = 0;
  let hearingWorkletUrl = "";

  function stopHearingTap() {
    hearingTapGen += 1;
    if (!hearingTap) return;
    try {
      hearingTap.proc.port.onmessage = null;
      hearingTap.proc.disconnect();
    } catch (_) {}
    try {
      hearingTap.src.disconnect();
    } catch (_) {}
    try {
      hearingTap.mute.disconnect();
    } catch (_) {}
    try {
      hearingTap.ctx.close();
    } catch (_) {}
    hearingTap = null;
  }

  function hearingWorkletModule() {
    if (!hearingWorkletUrl) {
      hearingWorkletUrl = URL.createObjectURL(new Blob([HEARING_WORKLET], { type: "application/javascript" }));
    }
    return hearingWorkletUrl;
  }

  function downsample(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = fromRate / toRate;
    const n = Math.max(1, Math.floor(input.length / ratio));
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = input[Math.min(input.length - 1, Math.floor(i * ratio))];
    return out;
  }

  function encodeWav16k(float32) {
    const n = float32.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buf);
    const writeStr = (offset, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + n * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 16000 * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, n * 2, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
    const bytes = new Uint8Array(buf);
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  function flushUtterance(chunks, ctxRate) {
    if (!chunks.length || !voiceCaptureAllowed()) return;
    const total = chunks.reduce((n, a) => n + a.length, 0);
    const merged = new Float32Array(total);
    let o = 0;
    for (let i = 0; i < chunks.length; i++) {
      merged.set(chunks[i], o);
      o += chunks[i].length;
    }
    const pcm = downsample(merged, ctxRate, 16000);
    if (pcm.length < 16000 * 0.4) return;
    const wavB64 = encodeWav16k(pcm);
    window.postMessage({ source: "lsa-page", type: "LSA_AUDIO_WAV", wavB64 }, "*");
  }

  async function startHearingTap() {
    const tracks = liveAudioTracks();
    if (!tracks.length) {
      window.postMessage(
        {
          source: "lsa-page",
          type: "LSA_SPEECH_ERROR",
          error: "Meet no está enviando tu mic. Prendé el micrófono de la llamada.",
        },
        "*"
      );
      return;
    }
    if (hearingTap && hearingTap.trackId === tracks[0].id) return;
    stopHearingTap();
    const gen = hearingTapGen;
    const trackId = tracks[0].id;
    const stream = new MediaStream([tracks[0]]);
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      await ctx.audioWorklet.addModule(hearingWorkletModule());
    } catch (err) {
      try {
        ctx.close();
      } catch (_) {}
      window.postMessage(
        {
          source: "lsa-page",
          type: "LSA_SPEECH_ERROR",
          error: "No se pudo abrir el tap de audio (AudioWorklet).",
        },
        "*"
      );
      return;
    }
    if (gen !== hearingTapGen || !hearingMode || hearingInput === "text") {
      try {
        ctx.close();
      } catch (_) {}
      return;
    }
    const src = ctx.createMediaStreamSource(stream);
    const proc = new AudioWorkletNode(ctx, "lsa-hearing", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
    });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    const chunks = [];
    let voiced = false;
    let silence = 0;
    const rmsOn = 0.018;
    const silenceEndMs = 1000;
    let announced = null;
    const noteMic = (open) => {
      if (announced === open) return;
      announced = open;
      window.postMessage(
        {
          source: "lsa-page",
          type: "LSA_SPEECH_UI",
          debug: open ? "Escuchando el mic de Meet" : "Micrófono de Meet silenciado",
        },
        "*"
      );
    };
    proc.port.onmessage = (ev) => {
      if (!hearingMode || hearingInput === "text") return;
      const open = voiceCaptureAllowed();
      noteMic(open);
      if (!open) {
        chunks.length = 0;
        voiced = false;
        silence = 0;
        return;
      }
      const input = ev.data;
      if (!input || !input.length) return;
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      if (rms >= rmsOn) {
        voiced = true;
        silence = 0;
        chunks.push(input);
      } else if (voiced) {
        chunks.push(input);
        silence += 1;
        const ms = (silence * input.length * 1000) / ctx.sampleRate;
        const dur = chunks.reduce((n, a) => n + a.length, 0) / ctx.sampleRate;
        if (ms >= silenceEndMs || dur > 4.5) {
          const copy = chunks.splice(0, chunks.length);
          voiced = false;
          silence = 0;
          flushUtterance(copy, ctx.sampleRate);
        }
      }
    };
    src.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
    ctx.resume().catch(() => {});
    if (!window.__lsaResumeTap) {
      window.__lsaResumeTap = true;
      window.addEventListener(
        "click",
        () => {
          if (hearingTap && hearingTap.ctx && hearingTap.ctx.state === "suspended") {
            hearingTap.ctx.resume().catch(() => {});
          }
        },
        true
      );
    }
    hearingTap = { ctx, src, proc, mute, trackId };
  }

  /**
   * Voice capture only keeps audio if Meet does not have the microphone muted.
   * The raw getUserMedia track still sounds even if the button is muted:
   * trust the button UI, and if it is missing, the call sender.
   */
  function voiceCaptureAllowed() {
    if (!hearingMode || hearingInput === "text") return false;
    if (meetUiMicMuted() === true) return false;
    const rtc = outboundKindLive("audio");
    if (rtc === false) return false;
    const tracks = liveAudioTracks();
    if (tracks.some((t) => t.muted === true)) return false;
    if (tracks.length && tracks.every((t) => t.enabled === false)) return false;
    if (meetUiMicMuted() === false) return true;
    return rtc === true;
  }

  /**
   * @param {MediaStreamConstraints|null|undefined} constraints
   * @returns {boolean} true si este pedido incluye (o implica) video.
   */
  function wantsVideo(constraints) {
    if (constraints == null) return true;
    if (Object.prototype.hasOwnProperty.call(constraints, "video")) {
      return Boolean(constraints.video);
    }
    if (constraints.audio) return false;
    return true;
  }

  /**
   * Split `text` into up to 3 lines that fit in `maxWidth`.
   * @param {CanvasRenderingContext2D} ctx
   * @param {string} text
   * @param {number} maxWidth
   * @returns {string[]}
   */
  function wrapLines(ctx, text, maxWidth) {
    const words = String(text).split(/\s+/).filter(Boolean);
    const lines = [];
    let current = "";
    for (const word of words) {
      const next = current ? current + " " + word : word;
      if (ctx.measureText(next).width > maxWidth && current) {
        lines.push(current);
        current = word;
      } else current = next;
    }
    if (current) lines.push(current);
    return lines.slice(-3);
  }

  function drawVideoFrame(ctx, video, w, h) {
    ctx.drawImage(video, 0, 0, w, h);
  }

  /** Camera cropped (cover) inside a rectangle, without stretching. */
  function drawCover(ctx, video, x, y, w, h) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    const scale = Math.max(w / vw, h / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.drawImage(video, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
    ctx.restore();
  }

  function avatarBusy() {
    if (!avatarClip || !window.LsaAvatar) return false;
    return performance.now() < avatarClip.t0 + window.LsaAvatar.durationMs(avatarClip);
  }

  function holdCaptionForClip(play) {
    const ms = Math.max(SUBTITLE_HOLD_MS, window.LsaAvatar.durationMs(play) + 600);
    if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
    window.__lsaCaptionTimer = setTimeout(() => {
      if (!avatarBusy() && !avatarQueue.length) caption.spanish = "";
    }, ms);
  }

  function beginAvatarClip(clip, line) {
    if (!window.LsaAvatar || !clip) return;
    avatarClip = window.LsaAvatar.start(clip);
    if (line) caption.spanish = line;
    holdCaptionForClip(avatarClip);
  }

  function enqueueAvatarClip(clip, line) {
    if (!clip || !clip.frames || !clip.frames.length) return;
    if (avatarBusy()) {
      if (avatarQueue.length >= AVATAR_QUEUE_MAX) avatarQueue.shift();
      avatarQueue.push({ clip, line: line || "" });
      return;
    }
    beginAvatarClip(clip, line);
  }

  function pumpAvatarQueue() {
    if (avatarBusy()) return;
    avatarClip = null;
    const next = avatarQueue.shift();
    if (next) beginAvatarClip(next.clip, next.line);
  }

  function drawSkeletonPanel(ctx, w, h) {
    if (!window.LsaAvatar) return;
    const half = Math.floor(w / 2);
    const panelW = w - half;
    let view = window.LsaAvatar.sample(avatarClip, performance.now());
    if (view && view.done) {
      pumpAvatarQueue();
      view = window.LsaAvatar.sample(avatarClip, performance.now());
    }
    window.LsaAvatar.draw(ctx, view, half, 0, panelW, h);
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.fillRect(half, 0, 1, h);
  }

  /** How many X-flips the element has (transform, scale, rotateY). */
  function flipCount(el) {
    let count = 0;
    let cs;
    try {
      cs = getComputedStyle(el);
    } catch (_) {
      return 0;
    }
    const scale = (cs.scale || "none").trim();
    if (scale && scale !== "none" && parseFloat(scale.split(/\s+/)[0]) < 0) count += 1;
    const transform = cs.transform || "";
    const m3 = transform.match(/matrix3d\(\s*(-?[\d.eE+-]+)(?:,\s*[^,]+){4},\s*(-?[\d.eE+-]+)/);
    if (m3) {
      if (parseFloat(m3[1]) < 0) count += 1;
    } else {
      const match = transform.match(/matrix\(\s*(-?[\d.eE+-]+)/);
      if (match && parseFloat(match[1]) < 0) count += 1;
    }
    if (/rotateY\(\s*180/.test(transform) || /rotate3d\(\s*0\s*,\s*1/.test(transform)) count += 1;
    return count;
  }

  function isOurPreviewVideo(video) {
    if (!video || video.id === "lsa-real-cam" || !pipeline || !pipeline.out) return false;
    const src = video.srcObject;
    if (!src || !src.getVideoTracks) return false;
    if (src === pipeline.out || src === pipeline.real) return true;
    const ours = new Set(pipeline.out.getVideoTracks().map((t) => t.id));
    if (pipeline.real) pipeline.real.getVideoTracks().forEach((t) => ours.add(t.id));
    return src.getVideoTracks().some((t) => ours.has(t.id));
  }

  function clearPreviewFix(video) {
    if (!video) return;
    const mine = video.dataset.lsaFix || "";
    if (mine === "scale") video.style.removeProperty("scale");
    if (mine === "transform") video.style.removeProperty("transform");
    if (mine) video.dataset.lsaFix = "";
  }

  /**
   * Meet espeja la miniatura local. El canvas (la llamada) va derecho.
   * Un scaleX extra en ESE video cancela el selfie; quien recibe no lo ve.
   */
  function straightenPreview(video) {
    if (!video) return;
    clearPreviewFix(video);
    let flips = flipCount(video);
    let el = video.parentElement;
    const seen = new Set();
    while (el && el !== document.documentElement && !seen.has(el)) {
      seen.add(el);
      flips += flipCount(el);
      el = el.parentElement || (el.getRootNode && el.getRootNode().host) || null;
    }
    if (flips % 2 === 0) return;
    const scaleNow = (getComputedStyle(video).scale || "none").trim();
    const scaleFlipped = scaleNow !== "none" && parseFloat(scaleNow.split(/\s+/)[0]) < 0;
    if (scaleFlipped) {
      video.style.setProperty("transform", "scaleX(-1)", "important");
      video.dataset.lsaFix = "transform";
    } else {
      video.style.setProperty("scale", "-1 1", "important");
      video.dataset.lsaFix = "scale";
    }
  }

  /** Left: camera. Right: skeleton. Both halves fill the full height. */
  function drawSplit(ctx, video, w, h) {
    const half = Math.floor(w / 2);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, w, h);
    drawCover(ctx, video, 0, 0, half, h);
    drawSkeletonPanel(ctx, w, h);
  }

  function drawLandmarks(ctx, w, h) {
    if (!showLandmarks || !landmarks) return;
    const drawSet = (pts, edges, color) => {
      if (!pts || !pts.length) return;
      const mapped = pts;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(2, Math.round(w / 280));
      ctx.beginPath();
      for (const [a, b] of edges) {
        const pa = mapped[a];
        const pb = mapped[b];
        if (!pa || !pb) continue;
        if ((pa.visibility || 1) < 0.4 || (pb.visibility || 1) < 0.4) continue;
        if (pa.x < -0.05 || pa.x > 1.05 || pb.x < -0.05 || pb.x > 1.05) continue;
        ctx.moveTo(pa.x * w, pa.y * h);
        ctx.lineTo(pb.x * w, pb.y * h);
      }
      ctx.stroke();
      const r = Math.max(2, Math.round(w / 160));
      for (const p of mapped) {
        if (!p || (p.visibility || 1) < 0.4) continue;
        if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) continue;
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    ctx.save();
    drawSet(landmarks.pose, POSE_EDGES, "rgba(91, 203, 232, 0.95)");
    drawSet(landmarks.left_hand, HAND_EDGES, "rgba(232, 154, 148, 0.95)");
    drawSet(landmarks.right_hand, HAND_EDGES, "rgba(234, 244, 250, 0.95)");
    ctx.restore();
  }

  /**
   * Caption at the bottom, upright on the canvas. Meet may mirror the hearing
   * user's thumbnail; the receiver sees this drawing without that CSS.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   */
  function drawSubtitles(ctx, w, h) {
    const text = caption.spanish ? String(caption.spanish).normalize("NFC") : "";
    if (!text) return;
    const fontSize = Math.max(18, Math.round(h * 0.042));
    ctx.save();
    ctx.font = `700 ${fontSize}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const maxTextW = Math.round(w * 0.88);
    const lines = wrapLines(ctx, text, maxTextW);
    const lineH = Math.round(fontSize * 1.22);
    const boxPadX = Math.round(fontSize * 0.85);
    const boxPadY = Math.round(fontSize * 0.42);
    const boxH = lines.length * lineH + boxPadY * 2;
    const boxW = Math.min(Math.round(w * 0.94), maxTextW + boxPadX * 2);
    const boxX = Math.round((w - boxW) / 2);
    const boxY = Math.round(h - boxH - h * 0.045);
    ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
    ctx.fillRect(boxX, boxY, boxW, boxH);
    ctx.lineWidth = Math.max(2, Math.round(fontSize / 11));
    ctx.strokeStyle = "rgba(0,0,0,0.9)";
    ctx.fillStyle = "#fff";
    const midX = w / 2;
    lines.forEach((line, i) => {
      const y = boxY + boxPadY + i * lineH + lineH / 2;
      ctx.strokeText(line, midX, y);
      ctx.fillText(line, midX, y);
    });
    ctx.restore();
  }

  /**
   * Release the real camera and the canvas. Otherwise Meet thinks it is blocked
   * (the device is still busy, or we return a silent canvas track).
   * @param {boolean} releaseCamera Also stop the native tracks.
   */
  function stopPipeline(releaseCamera) {
    if (!pipeline) return;
    const current = pipeline;
    current.stopped = true;
    current.running = false;
    pipeline = null;
    try {
      if (current.out) {
        current.out.getTracks().forEach((t) => {
          const mixTrack =
            callMix && callMix.dest.stream.getTracks().some((m) => m.id === t.id);
          if (!mixTrack) t.stop();
        });
      }
    } catch (_) {}
    if (releaseCamera) {
      try {
        current.real.getTracks().forEach((t) => t.stop());
      } catch (_) {}
    }
    try {
      const el = document.getElementById("lsa-real-cam");
      if (el) {
        el.srcObject = null;
        el.remove();
      }
    } catch (_) {}
    landmarks = null;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: false }, "*");
  }

  /** Meet sees the native stream again; do not stop those tracks. */
  function restoreNativeCamera() {
    if (!pipeline) return;
    const real = pipeline.real;
    const realTrack = real && real.getVideoTracks().find((t) => t.readyState === "live");
    document.querySelectorAll("video").forEach((v) => {
      if (v.id === "lsa-real-cam") return;
      if (v.srcObject === pipeline.out && real) v.srcObject = real;
    });
    if (realTrack) {
      peerConnections.forEach((pc) => {
        try {
          pc.getSenders().forEach((sender) => {
            if (sender.track && sender.track.kind === "video" && sender.track !== realTrack) {
              sender.replaceTrack(realTrack).catch(() => {});
            }
          });
        } catch (_) {}
      });
    }
    stopPipeline(false);
  }
  /**
   * Create hidden video `#lsa-real-cam` + canvas that Meet sees as “camera”.
   * The stream must be returned NOW: if getUserMedia takes several seconds,
   * Meet shows “camera blocked”. DroidCam can paint later; a black canvas
   * at 24 fps is enough so the track is not silent.
   *
   * @param {MediaStream} realStream Native `getUserMedia` result.
   */
  function createPipeline(realStream) {
    if (pipeline && pipeline.real === realStream && pipeline.alive()) {
      const live = existingStream();
      if (live) return pipeline;
    }
    if (pipeline) stopPipeline(pipeline.real !== realStream);
    const track = realStream.getVideoTracks()[0];
    const settings = (track && track.getSettings && track.getSettings()) || {};
    const video = document.createElement("video");
    video.id = "lsa-real-cam";
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.setAttribute("playsinline", "");
    video.setAttribute("muted", "");
    video.srcObject = realStream;
    video.style.cssText =
      "position:fixed;left:-2000px;top:0;width:640px;height:480px;opacity:0.01;pointer-events:none;";
    (document.body || document.documentElement).appendChild(video);
    video.play().catch(() => {});

    const canvas = document.createElement("canvas");
    canvas.width = settings.width || video.videoWidth || 640;
    canvas.height = settings.height || video.videoHeight || 480;
    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const handle = {
      real: realStream,
      out: null,
      stopped: false,
      running: true,
      alive() {
        return Boolean(
          !this.stopped &&
            realStream.getVideoTracks().some((t) => t.readyState === "live")
        );
      },
    };

    const out = canvas.captureStream(24);
    const capTrack = out.getVideoTracks()[0];
    if (capTrack) {
      if (capTrack.requestFrame) capTrack.requestFrame();
      capTrack.addEventListener("ended", () => {
        if (handle.stopped || pipeline !== handle) return;
        stopPipeline(true);
      });
    }
    let previewTick = 0;
    let fpsFrames = 0;
    let fpsT0 = performance.now();

    const draw = () => {
      if (!handle.running) return;
      fpsFrames += 1;
      const fpsSpan = performance.now() - fpsT0;
      if (fpsSpan >= 1000) {
        window.postMessage(
          {
            source: "lsa-page",
            type: "LSA_FPS",
            fps: (fpsFrames * 1000) / fpsSpan,
            captureStream: 24,
          },
          "*"
        );
        fpsFrames = 0;
        fpsT0 = performance.now();
      }
      const camOn = publishCamOpen();
      if (!camOn) {
        if (!video.paused) video.pause();
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (capTrack && capTrack.requestFrame) capTrack.requestFrame();
        requestAnimationFrame(draw);
        return;
      }
      if (video.paused) video.play().catch(() => {});
      if (video.readyState >= 2 && video.videoWidth) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        if (enabled && hearingMode && window.LsaAvatar) {
          drawSplit(ctx, video, canvas.width, canvas.height);
        } else {
          drawVideoFrame(ctx, video, canvas.width, canvas.height);
          if (enabled && showLandmarks) drawLandmarks(ctx, canvas.width, canvas.height);
        }
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (enabled && hearingMode && window.LsaAvatar) drawSkeletonPanel(ctx, canvas.width, canvas.height);
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height);
      }
      if (enabled && previewTick++ % 2 === 0) {
        document.querySelectorAll("video").forEach((v) => {
          if (isOurPreviewVideo(v)) straightenPreview(v);
        });
      }
      if (capTrack && capTrack.requestFrame) capTrack.requestFrame();
      requestAnimationFrame(draw);
    };
    draw();
    if (hearingMode) {
      realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    } else {
      const mixed = ensureCallMix(realStream);
      const mixedTrack = mixed && mixed.getAudioTracks()[0];
      const ctxRunning = callMix && callMix.ctx.state === "running";
      if (ctxRunning && mixedTrack) out.addTrack(mixedTrack);
      else realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    }
    handle.out = out;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: true }, "*");
    syncCanvasToMeet();
    if (hearingMode) syncHearingCapture();
    return handle;
  }

  /**
   * If Meet already took native DroidCam, Spanish painted on the canvas is not seen.
   * Put the canvas on the preview and on the track sent to the call.
   */
  function syncCanvasToMeet() {
    if (!pipeline || !pipeline.out) return;
    if (!meetCamOpen()) return;
    const canvasTrack = pipeline.out.getVideoTracks()[0];
    if (!canvasTrack || canvasTrack.readyState !== "live") return;
    document.querySelectorAll("video").forEach((v) => {
      if (v.id === "lsa-real-cam") return;
      const src = v.srcObject;
      if (!(src instanceof MediaStream)) return;
      const fromReal =
        pipeline.real &&
        src.getVideoTracks().some((t) => pipeline.real.getVideoTracks().indexOf(t) !== -1);
      if (src !== pipeline.out && (fromReal || src === pipeline.real)) v.srcObject = pipeline.out;
      if (isOurPreviewVideo(v)) straightenPreview(v);
    });
    peerConnections.forEach((pc) => {
      try {
        pc.getSenders().forEach((sender) => {
          if (sender.track && sender.track.kind === "video" && sender.track !== canvasTrack) {
            sender.replaceTrack(canvasTrack).catch(() => {});
          }
          const mixedAudio = pipeline.out.getAudioTracks()[0];
          const rawIds = pipeline.real ? pipeline.real.getAudioTracks().map((t) => t.id) : [];
          if (
            mixedAudio &&
            sender.track &&
            sender.track.kind === "audio" &&
            sender.track !== mixedAudio &&
            rawIds.indexOf(sender.track.id) !== -1
          ) {
            sender.replaceTrack(mixedAudio).catch(() => {});
          }
        });
      } catch (_) {}
    });
  }

  function wrapExistingMeetCamera() {
    if (!enabled) return;
    if (pipeline && pipeline.alive()) {
      syncCanvasToMeet();
      return;
    }
    if (pipeline && pipeline.real && pipeline.real.getVideoTracks().some((t) => t.readyState === "live")) {
      try {
        pipeline = createPipeline(pipeline.real);
        syncCanvasToMeet();
        return;
      } catch (_) {}
    }
    const videos = document.querySelectorAll("video");
    for (let i = 0; i < videos.length; i++) {
      const v = videos[i];
      if (v.id === "lsa-real-cam") continue;
      const src = v.srcObject;
      if (!(src instanceof MediaStream)) continue;
      if (!src.getVideoTracks().some((t) => t.readyState === "live")) continue;
      try {
        pipeline = createPipeline(src);
        v.srcObject = pipeline.out;
        syncCanvasToMeet();
        return;
      } catch (_) {}
    }
  }

  /**
   * Reuse the canvas. Never return raw DroidCam: there are no captions there.
   */
  function existingStream() {
    if (!enabled || !pipeline || pipeline.stopped || !pipeline.alive()) return null;
    const canvasTrack = pipeline.out && pipeline.out.getVideoTracks()[0];
    if (canvasTrack && canvasTrack.readyState === "live") return pipeline.out;
    return null;
  }

  function reuseHeldCamera() {
    if (!enabled || !pipeline || !pipeline.real) return null;
    const live = pipeline.real.getVideoTracks().some((t) => t.readyState === "live");
    if (!live) return null;
    try {
      pipeline = createPipeline(pipeline.real);
      return pipeline.out;
    } catch (_) {
      return null;
    }
  }

  MediaDevices.prototype.getUserMedia = function (constraints) {
    if (!wantsVideo(constraints)) {
      return origGetUserMedia.call(this, constraints).then((stream) => {
        extraAudio = stream;
        if (!hearingMode) ensureCallMix(stream);
        if (hearingMode) syncHearingCapture();
        const mixed = callMix && callMix.dest.stream;
        const running = callMix && callMix.ctx.state === "running";
        return !hearingMode && running && mixed ? mixed : stream;
      });
    }
    if (!enabled) {
      return origGetUserMedia.call(this, constraints);
    }
    const reuse = existingStream() || reuseHeldCamera();
    if (reuse) return Promise.resolve(reuse);
    return origGetUserMedia.call(this, constraints).then((real) => {
      try {
        pipeline = createPipeline(real);
        return pipeline.out;
      } catch (_) {
        return real;
      }
    });
  };
})();
