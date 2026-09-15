/**
 * Hook de getUserMedia en el mundo MAIN.
 * Con LSA off deja el stream nativo (si no, Meet dice cámara bloqueada).
 * Con LSA on Meet recibe un canvas: subtítulo espejado por el CSS mirror.
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
  let showSubtitles = true;
  let showLandmarks = false;
  let landmarks = null;
  let caption = { spanish: "", glosses: "" };
  let pipeline = null;

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "lsa-ext") return;
    if (data.type === "LSA_SET_ENABLED") {
      enabled = Boolean(data.enabled);
      if (!enabled) stopPageSpeech();
      syncCanvasToMeet();
    }
    if (data.type === "LSA_PREFS") {
      if (typeof data.showSubtitles === "boolean") showSubtitles = data.showSubtitles;
      if (typeof data.showLandmarks === "boolean") {
        showLandmarks = data.showLandmarks;
        if (!showLandmarks) landmarks = null;
      }
    }
    if (data.type === "LSA_LANDMARKS") {
      landmarks = data.landmarks || null;
    }
    if (data.type === "LSA_SPEECH_START") startPageSpeech();
    if (data.type === "LSA_SPEECH_STOP") stopPageSpeech();
    if (data.type === "LSA_CAPTION") {
      if (typeof data.spanish === "string" && data.spanish.length) {
        caption.spanish = data.spanish;
        if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
        window.__lsaCaptionTimer = setTimeout(() => {
          caption.spanish = "";
        }, SUBTITLE_HOLD_MS);
      }
      if (typeof data.glosses === "string") caption.glosses = data.glosses;
    }
  });

  let recognizer = null;

  function stopPageSpeech() {
    if (!recognizer) return;
    recognizer.onend = null;
    try {
      recognizer.stop();
    } catch (_) {}
    recognizer = null;
  }

  function startPageSpeech() {
    stopPageSpeech();
    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Rec) {
      window.postMessage(
        { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: "Este Chrome no tiene reconocimiento de voz" },
        "*"
      );
      return;
    }
    recognizer = new Rec();
    recognizer.lang = "es-AR";
    recognizer.continuous = true;
    recognizer.interimResults = true;
    recognizer.onresult = (event) => {
      let interim = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += piece;
        else interim += piece;
      }
      const spanish = (finalText || interim).trim();
      if (!spanish) return;
      caption.spanish = spanish;
      if (window.__lsaCaptionTimer) clearTimeout(window.__lsaCaptionTimer);
      window.__lsaCaptionTimer = setTimeout(() => {
        caption.spanish = "";
      }, SUBTITLE_HOLD_MS);
      window.postMessage(
        {
          source: "lsa-page",
          type: "LSA_SPEECH",
          spanish,
          final: Boolean(finalText),
        },
        "*"
      );
    };
    recognizer.onerror = (event) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      window.postMessage(
        { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: event.error || "error" },
        "*"
      );
    };
    recognizer.onend = () => {
      if (recognizer) {
        try {
          recognizer.start();
        } catch (_) {}
      }
    };
    try {
      recognizer.start();
    } catch (_) {
      window.postMessage(
        { source: "lsa-page", type: "LSA_SPEECH_ERROR", error: "No se pudo iniciar el micrófono de transcripción" },
        "*"
      );
    }
  }

  /**
   * @param {MediaStreamConstraints|null|undefined} constraints
   * @returns {boolean} true si este pedido incluye (o implica) video.
   */
  function wantsVideo(constraints) {
    if (constraints == null) return true;
    if (constraints.video === undefined) return true;
    return Boolean(constraints.video);
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
   * Subtítulo abajo, espejado para el CSS mirror de Meet.
   * Meet puede recortar bordes si el encuadre automático sigue activo.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   */
  function drawSubtitles(ctx, w, h) {
    const text = caption.spanish;
    if (!text) return;
    const fontSize = Math.max(18, Math.round(h * 0.042));
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
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

    const draw = () => {
      if (!handle.running) return;
      if (video.paused) video.play().catch(() => {});
      if (video.readyState >= 2 && video.videoWidth) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        drawVideoFrame(ctx, video, canvas.width, canvas.height);
        if (enabled && showLandmarks) drawLandmarks(ctx, canvas.width, canvas.height);
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (enabled && showSubtitles) drawSubtitles(ctx, canvas.width, canvas.height);
      }
      if (capTrack && capTrack.requestFrame) capTrack.requestFrame();
      requestAnimationFrame(draw);
    };
    draw();
    realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    handle.out = out;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: true }, "*");
    syncCanvasToMeet();
    return handle;
  }

  /**
   * Si Meet ya tomó DroidCam nativa, el español pintado en el canvas no se ve.
   * Ponemos el canvas en el preview y en el track que se manda a la llamada.
   */
  function syncCanvasToMeet() {
    if (!pipeline || !pipeline.out) return;
    const canvasTrack = pipeline.out.getVideoTracks()[0];
    if (!canvasTrack || canvasTrack.readyState !== "live") return;
    document.querySelectorAll("video").forEach((v) => {
      if (v.id === "lsa-real-cam") return;
      const src = v.srcObject;
      if (!(src instanceof MediaStream)) return;
      if (src === pipeline.out) return;
      const fromReal =
        pipeline.real &&
        src.getVideoTracks().some((t) => pipeline.real.getVideoTracks().indexOf(t) !== -1);
      if (fromReal || src === pipeline.real) v.srcObject = pipeline.out;
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
      return origGetUserMedia.call(this, constraints);
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
