/**
 * Hook de getUserMedia en el mundo MAIN.
 * Con LSA off deja el stream nativo (si no, Meet dice cámara bloqueada).
 * Con LSA on Meet recibe un canvas (captureStream).
 * Modo oyente: mitad izquierda cámara, mitad derecha esqueleto, misma altura.
 */
(function () {
  if (window.__lsaGumHooked) return;
  window.__lsaGumHooked = true;

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
  let pipeline = null;
  let extraAudio = null;

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "lsa-ext") return;
    if (data.type === "LSA_SET_ENABLED") {
      enabled = Boolean(data.enabled);
      hearingMode = Boolean(data.enabled && data.hearing);
      if (!hearingMode) {
        stopHearingTap();
        avatarClip = null;
      }
      if (hearingMode) {
        wrapExistingMeetCamera();
        syncHearingCapture();
        [300, 1000, 2500].forEach((ms) => {
          setTimeout(() => {
            if (hearingMode) {
              wrapExistingMeetCamera();
              syncHearingCapture();
            }
          }, ms);
        });
      }
      syncCanvasToMeet();
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
    if (data.type === "LSA_AVATAR" && data.clip && data.clip.ok && window.LsaAvatar) {
      avatarClip = window.LsaAvatar.start(data.clip);
      const ms = Math.max(SUBTITLE_HOLD_MS, window.LsaAvatar.durationMs(avatarClip) + 600);
      if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
      window.__lsaCaptionTimer = setTimeout(() => {
        caption.spanish = "";
      }, ms);
    }
    if (data.type === "LSA_CAPTION") {
      // Solo `line`: oyente manda glosas del semántico; sordo manda el español.
      // Nunca usar `glosses` crudas del clasificador.
      const line = typeof data.line === "string" ? data.line.trim() : "";
      if (line) {
        caption.spanish = line;
        if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
        window.__lsaCaptionTimer = setTimeout(() => {
          caption.spanish = "";
        }, SUBTITLE_HOLD_MS);
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
   * true = el botón de Meet dice que el micrófono está silenciado.
   * false = el botón lo muestra prendido. null = no se encontró el control.
   * “Desactivar / silenciar” es la acción: el mic está prendido.
   * “Activar el micrófono” también cuenta (hay un “el” en el medio).
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

  /** true = cámara apagada en la UI de Meet. */
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
   * null = todavía no hay senders (antes de unirse).
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
   * Por voz solo se guarda audio si Meet no tiene el micrófono silenciado.
   * El track crudo de getUserMedia sigue sonando aunque el botón esté en mute:
   * manda la UI del botón y, si no aparece, el sender de la llamada.
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
   * Parte `text` en hasta 3 líneas que entren en `maxWidth`.
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

  /** Cámara recortada (cover) dentro de un rectángulo, sin deformar. */
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

  function drawSkeletonPanel(ctx, w, h) {
    if (!window.LsaAvatar) return;
    const half = Math.floor(w / 2);
    const panelW = w - half;
    const view = window.LsaAvatar.sample(avatarClip, performance.now());
    if (view && view.done) avatarClip = null;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    window.LsaAvatar.draw(ctx, view, 0, 0, panelW, h);
    ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.fillRect(half, 0, 1, h);
  }

  /** Cuántos espejos en X tiene el elemento (transform, scale, rotateY). */
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
    const ours = new Set(pipeline.out.getVideoTracks().map((t) => t.id));
    return src.getVideoTracks().some((t) => ours.has(t.id));
  }

  /**
   * Meet espeja solo tu miniatura. El canvas: cámara y subtítulo derechos;
   * esqueleto ya espejado al dibujar. Este contrarresto es CSS del video local;
   * quien recibe no lo ve. Meet suele clonar el MediaStream: se matchea por
   * id de track, no por identidad de `srcObject`.
   */
  function straightenPreview(video) {
    if (!video) return;
    const mine = video.dataset.lsaFix || "";
    if (mine === "scale") video.style.removeProperty("scale");
    if (mine === "transform") video.style.removeProperty("transform");
    let flips = flipCount(video);
    for (let el = video.parentElement; el && el !== document.documentElement; el = el.parentElement) {
      flips += flipCount(el);
    }
    if (flips % 2 === 0) {
      video.dataset.lsaFix = "";
      return;
    }
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

  /** Izquierda: cámara. Derecha: esqueleto. Las dos mitades ocupan toda la altura. */
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
   * Subtítulo abajo. En modo sordo se espeja para el CSS mirror del preview local.
   * En modo oyente queda derecho: lo lee quien recibe el video.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   * @param {boolean} forReceiver
   */
  function drawSubtitles(ctx, w, h, forReceiver) {
    const text = caption.spanish;
    if (!text) return;
    const fontSize = Math.max(18, Math.round(h * 0.042));
    ctx.save();
    if (!forReceiver) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
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
   * Suelta la cámara real y el canvas. Si no, Meet cree que está bloqueada
   * (el dispositivo sigue ocupado o le devolvemos un track de canvas mudo).
   * @param {boolean} releaseCamera Parar también los tracks nativos.
   */
  function stopPipeline(releaseCamera) {
    if (!pipeline) return;
    pipeline.stopped = true;
    pipeline.running = false;
    try {
      pipeline.out.getTracks().forEach((t) => t.stop());
    } catch (_) {}
    if (releaseCamera) {
      try {
        pipeline.real.getTracks().forEach((t) => t.stop());
      } catch (_) {}
    }
    try {
      const el = document.getElementById("lsa-real-cam");
      if (el) {
        el.srcObject = null;
        el.remove();
      }
    } catch (_) {}
    pipeline = null;
    landmarks = null;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: false }, "*");
  }
  /**
   * Crea video oculto `#lsa-real-cam` + canvas que Meet ve como “cámara”.
   * Hay que devolver el stream YA: si getUserMedia tarda varios segundos,
   * Meet pone “cámara bloqueada”. DroidCam puede pintar después; un canvas
   * negro a 24 fps alcanza para que el track no quede mudo.
   *
   * @param {MediaStream} realStream Resultado nativo de `getUserMedia`.
   */
  function createPipeline(realStream) {
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
    if (capTrack && capTrack.requestFrame) capTrack.requestFrame();
    let previewTick = 0;

    const draw = () => {
      if (!handle.running) return;
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
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height, hearingMode);
      } else {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (enabled && hearingMode && window.LsaAvatar) drawSkeletonPanel(ctx, canvas.width, canvas.height);
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height, hearingMode);
      }
      if (hearingMode && previewTick++ % 8 === 0) {
        document.querySelectorAll("video").forEach((v) => {
          if (isOurPreviewVideo(v)) straightenPreview(v);
        });
      }
      if (capTrack && capTrack.requestFrame) capTrack.requestFrame();
      requestAnimationFrame(draw);
    };
    draw();
    realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    handle.out = out;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: true }, "*");
    syncCanvasToMeet();
    if (hearingMode) syncHearingCapture();
    return handle;
  }

  /**
   * Si Meet ya tomó DroidCam nativa, el español pintado en el canvas no se ve.
   * Ponemos el canvas en el preview y en el track que se manda a la llamada.
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
        });
      } catch (_) {}
    });
  }

  function wrapExistingMeetCamera() {
    if (pipeline && pipeline.alive()) {
      syncCanvasToMeet();
      return;
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
   * Reusa el canvas. Nunca devolvemos DroidCam cruda: ahí no hay subtítulos.
   */
  function existingStream() {
    if (!pipeline || pipeline.stopped || !pipeline.alive()) return null;
    const canvasTrack = pipeline.out && pipeline.out.getVideoTracks()[0];
    if (canvasTrack && canvasTrack.readyState === "live") return pipeline.out;
    return null;
  }

  MediaDevices.prototype.getUserMedia = function (constraints) {
    if (!wantsVideo(constraints)) {
      return origGetUserMedia.call(this, constraints).then((stream) => {
        extraAudio = stream;
        if (hearingMode) syncHearingCapture();
        return stream;
      });
    }
    const reuse = existingStream();
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
