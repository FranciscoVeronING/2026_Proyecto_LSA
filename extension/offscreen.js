/**
 * Offscreen: JPEG de Meet → sandbox (Holistic extrae landmarks) →
 * `capture.js` recorta la seña → `LsaApi.sign` (solo puntos, no video).
 */

const work = document.getElementById("work");
const workCtx = work.getContext("2d", { willReadFrequently: true });
const motionCanvas = document.getElementById("motion");
const motionCtx = motionCanvas.getContext("2d", { willReadFrequently: true });
const sandbox = document.getElementById("mp-sandbox");

let cfg = { ...FALLBACK_CFG };
let leftHanded = false;
let sandboxReady = false;
let busySign = false;
let signQueue = null;

function enqueueSign(frames) {
  if (busySign) {
    signQueue = frames;
    return;
  }
  busySign = true;
  emit({ status: "enviando", capturing: true });
  LsaApi.sign(frames)
    .then((s) => {
      if (s && s.accepted === false) {
        emit({
          debug: s.reason === "cooldown" ? "Esperá un segundo…" : "Seña corta, repetí",
          capturing: false,
        });
        return;
      }
      applyState(s);
      if (s && s.activity) engine.noteInferenceActivity();
    })
    .catch((err) => emit({ status: "error", error: err.message }))
    .finally(() => {
      busySign = false;
      const next = signQueue;
      signQueue = null;
      if (next) enqueueSign(next);
    });
}
let meetTabId = null;
let lastActivityPost = 0;
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

const engine = createCaptureEngine(
  () => cfg,
  {
    onSign(frames) {
      enqueueSign(frames);
    },
    onSigningActivity() {
      const now = Date.now();
      if (now - lastActivityPost < 250) return;
      lastActivityPost = now;
      LsaApi.activity().catch(() => {});
    },
    onUtterancePause() {
      emit({ status: "traduciendo" });
      return LsaApi.endUtterance()
        .then((s) => applyState(s))
        .catch((err) => emit({ status: "error", error: err.message }));
    },
  }
);

/**
 * Reenvía estado al service worker (HUD + subtítulos).
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
 * Aplica un snapshot del backend al HUD. Solo incluye `spanish` si vino en este tick
 * (si no, el subtítulo de Meet nunca caducaría).
 * @param {object|null|undefined} s Respuesta de `/sign`, `/session` o `/utterance/end`.
 */
function applyState(s) {
  if (!s) return;
  engine.markPending(Boolean(s.glosses && s.glosses.length));
  if (s.spanish) lastSpanish = s.spanish;
  const lastGloss = (s.gloss && String(s.gloss)) || (s.top3 && s.top3[0] && s.top3[0].gloss) || "";
  const payload = {
    status: s.semantic_busy ? "traduciendo" : lastSpanish ? "listo" : "escuchando",
    glosses: s.pending_text || "",
    lastGloss,
    top3: s.top3 || [],
  };
  if (s.closed && s.spanish) payload.spanish = s.spanish;
  emit(payload);
}

window.addEventListener("message", (event) => {
  const data = event.data;
  if (!data || !data.type) return;
  if (data.type === "ready") sandboxReady = true;
  if (data.type === "need-frame" || data.type === "error") {
    frameBusy = false;
    if (data.type === "error") emit({ status: "error", error: data.message });
    pumpFrame();
  }
  if (data.type === "landmarks") {
    frameBusy = false;
    pumpFrame();
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
 * Texto de estado del HUD, como máximo cada 350 ms.
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
      } else if (Date.now() - t0 > timeoutMs) {
        clearInterval(id);
        reject(new Error("MediaPipe no arrancó. Recargá la extensión."));
      }
    }, 50);
  });
}

/** Recarga `sandbox.html` si el iframe todavía no dijo `ready`. */
async function ensureSandbox() {
  if (sandboxReady && sandbox.contentWindow) return;
  sandboxReady = false;
  sandbox.src = "sandbox.html";
  await waitSandboxReady(20000);
}

/**
 * Health + config + sesión HTTP + WASM.
 * @param {number} tabId
 * @param {boolean} handed
 * @returns {Promise<void>}
 */
async function startSession(tabId, handed) {
  leftHanded = Boolean(handed);
  if (meetTabId === tabId && sandboxReady) {
    meetTabId = tabId;
    return;
  }
  meetTabId = tabId;
  const h = await LsaApi.health();
  if (!h.ok) throw new Error("El motor LSA no está encendido.");
  cfg = { ...FALLBACK_CFG, ...(await LsaApi.config()) };
  cfg.hands_frames_to_start = Math.min(3, Number(cfg.hands_frames_to_start) || 3);
  cfg.min_capture_frames = Math.min(6, Number(cfg.min_capture_frames) || 6);
  applyState(await LsaApi.session(leftHanded));
  await ensureSandbox();
  engine.reset();
  emit({ status: "escuchando", glosses: "", spanish: "" });
}

/** Resetea el recortador; no cierra Holistic (recargar WASM es caro). */
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
 * Decodifica un dataURL JPEG y se lo manda al sandbox (transferible ImageData).
 * Serializa con `frameBusy` para no saturar Holistic.
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
