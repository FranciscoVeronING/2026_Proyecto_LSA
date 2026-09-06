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

  function wantsVideo(constraints) {
    if (constraints == null) return true;
    if (constraints.video === undefined) return true;
    return Boolean(constraints.video);
  }

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

    let running = true;
    const draw = () => {
      if (!running) return;
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
    realStream.getAudioTracks().forEach((audioTrack) => out.addTrack(audioTrack));
    out.getVideoTracks().forEach((outTrack) => {
      outTrack.addEventListener("ended", () => {
        running = false;
      });
    });
    realStream.getVideoTracks().forEach((realTrack) => {
      realTrack.addEventListener("ended", () => {
        running = false;
      });
    });
    window.postMessage({ source: "lsa-page", type: "LSA_PIPELINE", live: true }, "*");
    return {
      real: realStream,
      out,
      stopped: false,
      alive() {
        return realStream.getVideoTracks().some((t) => t.readyState === "live");
      },
    };
  }

  function existingStream() {
    if (pipeline && pipeline.alive()) {
      try {
        return pipeline.out.clone();
      } catch (_) {
        return pipeline.out;
      }
    }
    return null;
  }

  MediaDevices.prototype.getUserMedia = function (constraints) {
    if (!wantsVideo(constraints)) {
      return origGetUserMedia.call(this, constraints);
    }
    const reuse = existingStream();
    if (reuse) return Promise.resolve(reuse);
    if (!enabled) {
      return origGetUserMedia.call(this, constraints);
    }
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
