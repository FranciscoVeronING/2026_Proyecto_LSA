const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const rec = document.getElementById("rec");
const dot = document.getElementById("dot");
const healthText = document.getElementById("health-text");
const glossesEl = document.getElementById("glosses");
const spanishEl = document.getElementById("spanish");
const top3El = document.getElementById("top3");
const capDebug = document.getElementById("cap-debug");
const motionCanvas = document.getElementById("motion");
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlay.getContext("2d");
const frameIn = document.getElementById("frame-in");
const frameCtx = frameIn.getContext("2d", { willReadFrequently: true });
const sandbox = document.getElementById("mp-sandbox");

let cfg = { ...FALLBACK_CFG };
let leftHanded = false;
let running = false;
let starting = false;
let stream = null;
let busySign = false;
let mpReady = false;
let sandboxReady = false;
let sending = false;
let loopGen = 0;
let lastResults = null;
let spanishTimer = null;
const SUBTITLE_HOLD_MS = 8000;
const inferHold = document.createElement("canvas");
const inferHoldCtx = inferHold.getContext("2d", { willReadFrequently: true });
const overlaySmoother = createLandmarkSmoother(0.45);

function applyState(s) {
  if (!s) return;
  glossesEl.textContent = s.pending_text || "(vacío)";
  if (s.semantic_busy) spanishEl.textContent = "Traduciendo…";
  else if (s.spanish) {
    spanishEl.textContent = s.spanish;
    if (spanishTimer) clearTimeout(spanishTimer);
    const shown = s.spanish;
    spanishTimer = setTimeout(() => {
      if (spanishEl.textContent === shown) spanishEl.textContent = "—";
    }, SUBTITLE_HOLD_MS);
  }
  if (s.top3 && s.top3.length) {
    top3El.textContent = s.top3
      .map((t) => `${t.gloss} (${Math.round(t.confidence * 100)}%)`)
      .join(" · ");
  }
  engine.markPending(Boolean(s.glosses && s.glosses.length));
}

let lastActivityPost = 0;
const engine = createCaptureEngine(
  () => cfg,
  {
    onSign(frames) {
      if (busySign) return;
      busySign = true;
      capDebug.textContent = `Enviando seña (${frames.length} frames)…`;
      LsaApi.sign(frames)
        .then((s) => {
          applyState(s);
          if (s.activity) engine.noteInferenceActivity();
        })
        .catch((err) => {
          healthText.textContent = err.message;
        })
        .finally(() => {
          busySign = false;
        });
    },
    onSigningActivity() {
      const now = Date.now();
      if (now - lastActivityPost < 250) return;
      lastActivityPost = now;
      LsaApi.activity().catch(() => {});
    },
    onUtterancePause() {
      capDebug.textContent = "Fin de oración (pausa)…";
      return LsaApi.endUtterance()
        .then((s) => {
          applyState(s);
        })
        .catch((err) => {
          healthText.textContent = err.message;
        });
    },
  }
);

document.querySelectorAll("#hand-toggle button").forEach((btn) => {
  btn.addEventListener("click", async () => {
    document.querySelectorAll("#hand-toggle button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    leftHanded = btn.dataset.hand === "left";
    try {
      applyState(await LsaApi.session(leftHanded));
    } catch (e) {
      healthText.textContent = e.message;
    }
  });
});

document.getElementById("btn-clear").addEventListener("click", async () => {
  try {
    applyState(await LsaApi.clearConversation());
    spanishEl.textContent = "—";
  } catch (e) {
    healthText.textContent = e.message;
  }
});

document.getElementById("btn-start").addEventListener("click", () => {
  if (starting) return;
  if (running) stopCamera();
  else startCamera();
});

async function ping() {
  try {
    const h = await LsaApi.health();
    if (!h.ok) throw new Error("no listo");
    dot.classList.add("on");
    if (!running) {
      const llm = h.semantic_ready ? "LLM lista" : "esperando LLM";
      healthText.textContent = `Motor OK · ${llm}`;
    }
    return true;
  } catch {
    dot.classList.remove("on");
    if (!running) healthText.textContent = "Motor apagado. Abrí la guía de instalación.";
    return false;
  }
}

function onResults(results) {
  mpReady = true;
  lastResults = results;
  const smoothed = overlaySmoother.apply(results);
  drawCameraAndLandmarks(overlayCtx, inferHold.width ? inferHold : video, smoothed);
  const info = engine.step({
    results,
    video,
    motionCtx,
    leftHanded,
  });
  rec.className = "rec-dot";
  if (info.recording && info.moving) rec.classList.add("live");
  else if (info.recording) rec.classList.add("still");
  else if (!info.handsPresent) rec.classList.add("lost");

  const pose = results.poseLandmarks ? "sí" : "no";
  const lh = results.leftHandLandmarks ? "sí" : "no";
  const rh = results.rightHandLandmarks ? "sí" : "no";
  capDebug.textContent =
    `Pose ${pose} · mano izq ${lh} · der ${rh} · ` +
    `buffer ${info.bufferLen}/${cfg.capture_buffer_size || 60} · ` +
    (info.recording ? (info.moving ? "SEÑANDO" : "corte (quieto)") : "esperando inicio");
}

window.addEventListener("message", (event) => {
  const data = event.data;
  if (!data || !data.type) return;
  if (data.type === "ready") {
    sandboxReady = true;
    capDebug.textContent = "MediaPipe listo en sandbox.";
  }
  if (data.type === "error") {
    sending = false;
    healthText.textContent = `MediaPipe: ${data.message}`;
    capDebug.textContent = data.message;
  }
  if (data.type === "need-frame" || data.type === "landmarks") {
    sending = false;
  }
  if (data.type === "landmarks") {
    onResults({
      poseLandmarks: data.pose,
      leftHandLandmarks: data.left_hand,
      rightHandLandmarks: data.right_hand,
    });
  }
});

function waitSandboxReady(timeoutMs) {
  if (sandboxReady) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const id = setInterval(() => {
      if (sandboxReady) {
        clearInterval(id);
        resolve();
      } else if (Date.now() - t0 > timeoutMs) {
        clearInterval(id);
        reject(new Error("El sandbox de MediaPipe no arrancó. Recargá la extensión."));
      }
    }, 50);
  });
}

async function ensureSandbox() {
  if (sandboxReady && sandbox.contentWindow) return;
  capDebug.textContent = "Esperando sandbox MediaPipe…";
  sandboxReady = false;
  sandbox.src = "sandbox.html";
  await waitSandboxReady(20000);
}

async function startCamera() {
  if (starting || running) return;
  starting = true;
  const btn = document.getElementById("btn-start");
  btn.disabled = true;
  try {
    const ok = await ping();
    if (!ok) return;
    cfg = { ...FALLBACK_CFG, ...(await LsaApi.config()) };
    applyState(await LsaApi.session(leftHanded));

    await ensureSandbox();
    if (!sandbox.contentWindow) {
      throw new Error("El sandbox de MediaPipe no está listo.");
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    running = true;
    mpReady = false;
    sending = false;
    lastResults = null;
    const gen = ++loopGen;
    btn.textContent = "Detener cámara";
    healthText.textContent = "Cámara on · esqueleto en sandbox";
    capDebug.textContent = "Esperando el primer frame de MediaPipe…";

    const loop = async () => {
      if (!running || gen !== loopGen) return;
      if (!sending && sandboxReady && sandbox.contentWindow && video.readyState >= 2 && video.videoWidth > 0) {
        sending = true;
        try {
          if (frameIn.width !== video.videoWidth || frameIn.height !== video.videoHeight) {
            frameIn.width = video.videoWidth;
            frameIn.height = video.videoHeight;
          }
          frameCtx.drawImage(video, 0, 0, frameIn.width, frameIn.height);
          if (inferHold.width !== frameIn.width || inferHold.height !== frameIn.height) {
            inferHold.width = frameIn.width;
            inferHold.height = frameIn.height;
          }
          inferHoldCtx.drawImage(frameIn, 0, 0);
          const imageData = frameCtx.getImageData(0, 0, frameIn.width, frameIn.height);
          if (!running || gen !== loopGen) {
            sending = false;
          } else {
            sandbox.contentWindow.postMessage(
              {
                type: "frame",
                width: imageData.width,
                height: imageData.height,
                buffer: imageData.data.buffer,
              },
              "*",
              [imageData.data.buffer]
            );
          }
        } catch (err) {
          sending = false;
          console.error(err);
          capDebug.textContent = `Frame: ${err.message || err}`;
        }
      }
      requestAnimationFrame(loop);
    };
    loop();

    setTimeout(() => {
      if (running && !mpReady) {
        capDebug.textContent = "No llegan landmarks del sandbox. Recargá la extensión (1.1.2).";
      }
    }, 10000);
  } catch (err) {
    healthText.textContent = err.message || String(err);
    capDebug.textContent = err.message || String(err);
    stopCamera();
  } finally {
    starting = false;
    btn.disabled = false;
    if (!running) btn.textContent = "Iniciar cámara";
  }
}

function stopCamera() {
  running = false;
  starting = false;
  loopGen += 1;
  sending = false;
  engine.reset();
  overlaySmoother.reset();
  lastResults = null;
  try {
    if (sandbox.contentWindow) sandbox.contentWindow.postMessage({ type: "stop" }, "*");
  } catch (_) {}
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  try {
    video.pause();
  } catch (_) {}
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width || 0, overlay.height || 0);
  document.getElementById("btn-start").textContent = "Iniciar cámara";
  document.getElementById("btn-start").disabled = false;
  capDebug.textContent = "Cámara detenida";
}

ping();
setInterval(ping, 4000);
