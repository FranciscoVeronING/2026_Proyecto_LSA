/**
 * Content script MAIN world (`document_start`) en meet.google.com.
 *
 * Meet pide la cámara con `getUserMedia`. Este archivo:
 * - Con LSA apagado: deja pasar el stream nativo (si no, Meet muestra “cámara bloqueada”).
 * - Con LSA encendido: toma el stream real, lo pinta en un canvas (español abajo
 *   espejado para compensar el CSS mirror de Meet) y le da a Meet `captureStream`.
 *
 * Habla con `meet.js` (mundo aislado) vía `window.postMessage`:
 *   lsa-ext → LSA_SET_ENABLED / LSA_CAPTION
 *   lsa-page → LSA_PIPELINE
 */
(function () {
  if (window.__lsaGumHooked) return;
  window.__lsaGumHooked = true;

  const origGetUserMedia = MediaDevices.prototype.getUserMedia;
  const SUBTITLE_HOLD_MS = 8000;
  let enabled = false;
  let caption = { spanish: "", glosses: "" };
  let pipeline = null;

  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "lsa-ext") return;
    if (data.type === "LSA_SET_ENABLED") {
      enabled = Boolean(data.enabled);
      if (!enabled) stopPipeline(true);
    }
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

  /**
   * Dibuja el español abajo. `scale(-1,1)` cancela el espejo local de Meet.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   */
  function drawSubtitles(ctx, w, h) {
    const text = caption.spanish;
    if (!text) return;
    const margin = Math.round(w * 0.06);
    const fontSize = Math.max(22, Math.round(h * 0.055));
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.font = `650 ${fontSize}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    const lines = wrapLines(ctx, text, w - margin * 2);
    const lineH = Math.round(fontSize * 1.2);
    const boxPad = Math.round(fontSize * 0.45);
    const boxH = lines.length * lineH + boxPad * 2;
    const boxY = h - boxH - Math.round(h * 0.04);
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(margin * 0.5, boxY, w - margin, boxH);
    ctx.lineWidth = Math.max(2, Math.round(fontSize / 12));
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.fillStyle = "#fff";
    lines.forEach((line, i) => {
      const y = boxY + boxPad + (i + 1) * lineH - Math.round(fontSize * 0.2);
      ctx.strokeText(line, w / 2, y);
      ctx.fillText(line, w / 2, y);
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
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: false }, "*");
  }
  /**
   * Crea video oculto `#lsa-real-cam` + canvas que Meet ve como “cámara”.
   * Pinta un frame ya: si el track de `captureStream` queda `muted`, Meet
   * muestra “cámara bloqueada”.
   *
   * @param {MediaStream} realStream Resultado nativo de `getUserMedia`.
   */
  function createPipeline(realStream) {
    const track = realStream.getVideoTracks()[0];
    const settings = (track && track.getSettings && track.getSettings()) || {};
    const video = document.createElement("video");
    video.id = "lsa-real-cam";
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.srcObject = realStream;
    video.style.cssText =
      "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;";
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
        const vt = this.out && this.out.getVideoTracks()[0];
        return Boolean(
          !this.stopped &&
            vt &&
            vt.readyState === "live" &&
            realStream.getVideoTracks().some((t) => t.readyState === "live")
        );
      },
    };

    const draw = () => {
      if (!handle.running) return;
      if (video.paused) video.play().catch(() => {});
      if (video.readyState >= 2 && video.videoWidth) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (enabled) drawSubtitles(ctx, canvas.width, canvas.height);
      }
      requestAnimationFrame(draw);
    };
    draw();

    const out = canvas.captureStream(30);
    if (video.readyState >= 2 && video.videoWidth) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    out.getVideoTracks().forEach((outTrack) => {
      outTrack.addEventListener("ended", () => {
        handle.running = false;
      });
    });
    realStream.getVideoTracks().forEach((realTrack) => {
      realTrack.addEventListener("ended", () => {
        handle.running = false;
      });
    });
    handle.out = out;
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: true }, "*");
    return handle;
  }

  /**
   * @param {HTMLVideoElement} video
   * @param {number} ms
   * @returns {Promise<void>}
   */
  function waitForVideo(video, ms) {
    if (video.videoWidth) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        video.removeEventListener("loadeddata", done);
        clearTimeout(tid);
        resolve();
      };
      const tid = setTimeout(done, ms);
      video.addEventListener("loadeddata", done);
    });
  }

  /** Reusa el canvas solo con LSA on y tracks vivos (no mudos). */
  function existingStream() {
    if (!enabled || !pipeline || pipeline.stopped || !pipeline.alive()) return null;
    const vt = pipeline.out.getVideoTracks()[0];
    if (!vt || vt.muted) return null;
    return pipeline.out;
  }

  MediaDevices.prototype.getUserMedia = function (constraints) {
    if (!wantsVideo(constraints)) {
      return origGetUserMedia.call(this, constraints);
    }
    if (!enabled) {
      stopPipeline(true);
      return origGetUserMedia.call(this, constraints);
    }
    const reuse = existingStream();
    if (reuse) return Promise.resolve(reuse);
    stopPipeline(true);
    return origGetUserMedia.call(this, constraints).then(async (real) => {
      try {
        pipeline = createPipeline(real);
        const video = document.getElementById("lsa-real-cam");
        if (video) await waitForVideo(video, 400);
        const vt = pipeline.out.getVideoTracks()[0];
        if (!vt || vt.readyState !== "live") return real;
        return pipeline.out;
      } catch (_) {
        return real;
      }
    });
  };
})();
