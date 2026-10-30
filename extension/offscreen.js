/**
 * Offscreen: Meet JPEG → sandbox (Holistic extracts landmarks) →
 * `capture.js` crops the sign → `LsaApi.sign` (points only, no video).
 */

const work = document.getElementById("work");
const workCtx = work.getContext("2d", { willReadFrequently: true });
const motionCanvas = document.getElementById("motion");
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });
const sandbox = document.getElementById("mp-sandbox");

let cfg = { ...FALLBACK_CFG };
let leftHanded = false;
let sandboxReady = false;
let sessionStarting = false;
let sandboxLastError = "";
let busySign = false;
let signQueue = null;
let holFrames = 0;
let holT0 = 0;
let lastHolisticFpsPost = 0;

function enqueueSign(frames, client) {
  if (busySign) {
    signQueue = { frames, client };
    return;
  }
  busySign = true;
  emit({ status: "enviando", capturing: true });
  const tPost = performance.now();
  LsaApi.sign(frames, client)
    .then((s) => {
      const post_ms = Math.round(performance.now() - tPost);
      if (s && s.accepted === false) {
        emit({
          debug: s.reason === "cooldown" ? "Esperá un segundo…" : "Seña corta, repetí",
          capturing: false,
        });
        return;
      }
      const clf = s && s.ms != null ? Math.round(s.ms) : "?";
      emit({
        debug: `Clasificador ${clf} ms · POST ${post_ms} ms`,
      });
      applyState(s);
      if (s && s.activity) engine.noteInferenceActivity();
    })
    .catch((err) => emit({ status: "error", error: err.message }))
    .finally(() => {
      busySign = false;
      const next = signQueue;
      signQueue = null;
      if (next) enqueueSign(next.frames, next.client);
    });
}
let meetTabId = null;
let lastActivityPost = 0;
let pendingSignEndMs = 0;
let prefs = { showLandmarks: false, outputMode: "subtitles" };
let lastLandmarkEmit = 0;
let lastDebugEmit = 0;

connectBackground();
LsaPrefs.get()
  .then((p) => {
    prefs = p;
  })
  .catch(() => {});
LsaPrefs.onChange((p) => {
  prefs = p;
});

let lastHolisticMs = 0;
let lastJpegAt = 0;
let lastJpegMs = 0;
const engine = createCaptureEngine(
  () => ({
    ...cfg,
    utterance_pause_sec: normalizePauseSec(prefs.utterancePauseSec),
  }),
  {
    onSign(frames, meta) {
      enqueueSign(frames, {
        capture_ms: meta && meta.capture_ms,
        n_raw: meta && meta.n_raw,
        n_sent: meta && meta.n_sent,
        time_to_record_ms: meta && meta.time_to_record_ms,
        holistic_ms: lastHolisticMs || undefined,
        jpeg_ms: lastJpegMs || undefined,
      });
    },
    onSigningActivity() {
      const now = Date.now();
      if (now - lastActivityPost < 250) return;
      lastActivityPost = now;
      LsaApi.activity().catch(() => {});
    },
    onUtterancePause(info) {
      emit({ status: "traduciendo" });
      const t0 = performance.now();
      const pause_ms = info && typeof info.pause_ms === "number" ? info.pause_ms : undefined;
      pendingSignEndMs =
        info && typeof info.sign_end_ms === "number"
          ? info.sign_end_ms
          : Date.now() - (typeof pause_ms === "number" ? pause_ms : 0);
      return LsaApi.endUtterance({ pause_ms })
        .then((s) => {
          const ms = s && s.ms != null ? Math.round(s.ms) : Math.round(performance.now() - t0);
          emit({ debug: `Traductor ${ms} ms (${(s && s.via) || "—"})` });
          applyState(s);
        })
        .catch((err) => emit({ status: "error", error: err.message }));
    },
  }
);

/**
 * Forward state to the service worker (HUD + captions).
 * @param {object} payload
 */
function emit(payload) {
  if (!meetTabId) return;
  chrome.runtime.sendMessage({
    type: "lsa-caption",
    tabId: meetTabId,
    ...payload,
  });
}

let lastSpanish = "";

/**
 * Apply a backend snapshot to the HUD. Only include `spanish` if it came in this tick
 * (otherwise the Meet caption would never expire).
 * @param {object|null|undefined} s Response from `/sign`, `/session`, or `/utterance/end`.
 */
function applyState(s) {
  if (!s) return;
  const pending = Array.isArray(s.glosses) ? s.glosses : [];
  if (s.closed) engine.markPending(false);
  else engine.markPending(Boolean(pending.length || s.pending_text));
  if (s.spanish) lastSpanish = s.spanish;
  const lastGloss = (s.gloss && String(s.gloss)) || (s.top3 && s.top3[0] && s.top3[0].gloss) || "";
  const payload = {
    status: s.semantic_busy ? "traduciendo" : lastSpanish ? "listo" : "escuchando",
    glosses: s.pending_text || "",
    pendingText: s.pending_text || "",
    lastGloss,
    top3: s.top3 || [],
  };
  if (typeof s.added === "boolean") {
    payload.glossAdded = s.added;
    payload.glossReject = s.reject_reason || "";
    payload.glossConfidence = typeof s.confidence === "number" ? s.confidence : null;
  }
  if (s.closed) {
    const line = String(s.spanish || lastSpanish || pending.join(" ")).trim();
    if (line) payload.spanish = line;
    if (pendingSignEndMs) payload.sign_end_ms = pendingSignEndMs;
  }
  emit(payload);
}

window.addEventListener("message", (event) => {
  const data = event.data;
  if (!data || !data.type) return;
  if (data.type === "ready") {
    sandboxReady = true;
    sandboxLastError = "";
  }
  if (data.type === "need-frame" || data.type === "error") {
    frameBusy = false;
    if (data.type === "error") {
      sandboxLastError = data.message || "Error en MediaPipe";
      emit({ status: "error", error: sandboxLastError });
    }
    pumpFrame();
  }
  if (data.type === "landmarks") {
    frameBusy = false;
    pumpFrame();
    if (typeof data.holistic_ms === "number") lastHolisticMs = data.holistic_ms;
    if (lastJpegAt) {
      lastJpegMs = Math.max(0, Math.round(performance.now() - lastJpegAt));
      lastJpegAt = 0;
    }
    const results = {
      poseLandmarks: data.pose,
      leftHandLandmarks: data.left_hand,
      rightHandLandmarks: data.right_hand,
    };
    const info = engine.step({
      results,
      video: work,
      motionCtx,
      leftHanded,
    });
    emitDebug(info);
    emitLandmarks(data);
    holFrames += 1;
    if (!holT0) holT0 = performance.now();
    const holSpan = performance.now() - holT0;
    if (holSpan >= 1000) {
      const holistic_fps = (holFrames * 1000) / holSpan;
      holFrames = 0;
      holT0 = performance.now();
      const now = Date.now();
      if (now - lastHolisticFpsPost >= 5000) {
        lastHolisticFpsPost = now;
        LsaApi.clientMetrics({ holistic_fps }).catch(() => {});
      }
    }
  }
});

function emitLandmarks(data) {
  if (!meetTabId || !prefs.showLandmarks) return;
  const now = Date.now();
  if (now - lastLandmarkEmit < 80) return;
  lastLandmarkEmit = now;
  chrome.runtime.sendMessage({
    type: "lsa-landmarks",
    tabId: meetTabId,
    landmarks: {
      pose: data.pose,
      left_hand: data.left_hand,
      right_hand: data.right_hand,
    },
  }).catch(() => {});
}
/**
 * HUD status text, at most every 350 ms.
 * @param {{ recording: boolean, bufferLen: number, handsPresent: boolean }} info
 */
function emitDebug(info) {
  const now = Date.now();
  if (now - lastDebugEmit < 350) return;
  lastDebugEmit = now;
  if (busySign) return;
  const max = cfg.capture_buffer_size || 60;
  let debug;
  if (info.recording) {
    debug = `Grabando seña ${info.bufferLen}/${max}`;
  } else if (info.handsPresent) {
    debug = "Manos a la vista";
  } else {
    debug = "Mostrá las manos para capturar";
  }
  emit({
    status: info.recording ? "grabando" : "escuchando",
    debug,
    capturing: Boolean(info.recording || info.handsPresent),
  });
}

/**
 * @param {number} timeoutMs
 * @returns {Promise<void>}
 */
function waitSandboxReady(timeoutMs) {
  if (sandboxReady && sandbox.contentWindow) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const id = setInterval(() => {
      if (sandboxReady && sandbox.contentWindow) {
        clearInterval(id);
        resolve();
      } else if (sandboxLastError) {
        clearInterval(id);
        reject(new Error(sandboxLastError));
      } else if (Date.now() - t0 > timeoutMs) {
        clearInterval(id);
        reject(
          new Error(
            "MediaPipe no arrancó. Faltan los WASM (.data/.wasm) en extension/vendor/mediapipe. Corré py -3 packaging/fetch_extension_assets.py y recargá la extensión."
          )
        );
      }
    }, 50);
  });
}

/** Reload `sandbox.html` if the iframe has not said `ready` yet. */
async function ensureSandbox() {
  if (sandboxReady && sandbox.contentWindow) return;
  sandboxReady = false;
  sandboxLastError = "";
  sandbox.src = "sandbox.html";
  await waitSandboxReady(20000);
}

/**
 * Health + config + HTTP session + WASM.
 * @param {number} tabId
 * @param {boolean} handed
 * @returns {Promise<void>}
 */
async function startSession(tabId, handed) {
  const nextHanded = Boolean(handed);
  const handedChanged = leftHanded !== nextHanded;
  leftHanded = nextHanded;
  if (meetTabId === tabId && (sandboxReady || sessionStarting)) {
    if (handedChanged) engine.forgetPrevHand();
    return;
  }
  sessionStarting = true;
  meetTabId = tabId;
  try {
    const h = await LsaApi.health();
    if (!h.ok) throw new Error("El motor LSA no está encendido.");
    cfg = { ...FALLBACK_CFG, ...(await LsaApi.config()) };
    cfg.hands_frames_to_start = 1;
    cfg.min_capture_frames = Math.min(2, Number(cfg.min_capture_frames) || 2);
    applyState(await LsaApi.session(leftHanded));
    await ensureSandbox();
    engine.reset();
    emit({ status: "escuchando", glosses: "" });
  } finally {
    sessionStarting = false;
  }
}

/** Reset the cropper; do not close Holistic (reloading WASM is expensive). */
function stopSession() {
  engine.reset();
  meetTabId = null;
  lastSpanish = "";
  try {
    if (sandbox.contentWindow) sandbox.contentWindow.postMessage({ type: "stop" }, "*");
  } catch (_) {}
}

let frameBusy = false;
let latestDataUrl = null;
const decodeImg = new Image();

/**
 * Decode a JPEG dataURL and send it to the sandbox (transferable ImageData).
 * Serialize with `frameBusy` so Holistic is not saturated.
 */
function pumpFrame() {
  if (frameBusy || !sandboxReady || !sandbox.contentWindow || !latestDataUrl) return;
  const url = latestDataUrl;
  latestDataUrl = null;
  frameBusy = true;
  decodeImg.onload = () => {
    try {
      if (work.width !== decodeImg.width || work.height !== decodeImg.height) {
        work.width = decodeImg.width;
        work.height = decodeImg.height;
      }
      workCtx.drawImage(decodeImg, 0, 0);
      const imageData = workCtx.getImageData(0, 0, work.width, work.height);
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
    } catch (err) {
      emit({ status: "error", error: String(err && err.message ? err.message : err) });
      frameBusy = false;
      pumpFrame();
    }
  };
  decodeImg.onerror = () => {
    frameBusy = false;
    pumpFrame();
  };
  decodeImg.src = "";
  decodeImg.src = url;
}

/**
 * @param {{ dataUrl?: string, buffer?: ArrayBuffer, width?: number, height?: number }} msg
 */
function handleFrame(msg) {
  if (msg.dataUrl) {
    latestDataUrl = msg.dataUrl;
    lastJpegAt = performance.now();
    pumpFrame();
    return;
  }
  if (!sandboxReady || !sandbox.contentWindow || !msg.buffer || !msg.width) return;
  const pixels = new Uint8ClampedArray(msg.buffer);
  const imageData = new ImageData(pixels, msg.width, msg.height);
  if (work.width !== msg.width || work.height !== msg.height) {
    work.width = msg.width;
    work.height = msg.height;
  }
  workCtx.putImageData(imageData, 0, 0);
  sandbox.contentWindow.postMessage(
    {
      type: "frame",
      width: msg.width,
      height: msg.height,
      buffer: imageData.data.buffer,
    },
    "*",
    [imageData.data.buffer]
  );
}

function handleControlMessage(msg, reply) {
  if (!msg || !msg.type) return false;
  if (msg.type === "lsa-prefs" && msg.prefs) {
    prefs = normalizeLsaPrefs(msg.prefs);
    return true;
  }
  if (msg.type === "lsa-meet-start") {
    startSession(msg.tabId, msg.leftHanded)
      .then(() => reply({ ok: true }))
      .catch((err) => reply({ ok: false, error: err.message }));
    return true;
  }
  if (msg.type === "lsa-meet-stop") {
    stopSession();
    reply({ ok: true });
    return true;
  }
  if (msg.type === "lsa-capture-reset") {
    if (engine) engine.reset();
    reply({ ok: true });
    return true;
  }
  if (msg.type === "lsa-frame" || msg.type === "lsa-offscreen-frame") {
    handleFrame(msg);
    reply({ ok: true });
    return true;
  }
  return false;
}

function connectBackground() {
  const port = chrome.runtime.connect({ name: "lsa-offscreen" });
  port.onMessage.addListener((msg) => {
    handleControlMessage(msg, (res) => {
      if (msg && msg.id) port.postMessage({ type: "lsa-ack", id: msg.id, ...res });
    });
  });
  port.onDisconnect.addListener(() => {
    setTimeout(connectBackground, 250);
  });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handled = handleControlMessage(msg, (res) => sendResponse(res));
  if (handled) return true;
});
