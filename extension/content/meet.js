/**
 * Meet, mundo aislado: HUD + JPEG al SW.
 * getUserMedia y el micrófono del oyente viven en inject-gum.js.
 */

const SUBTITLE_HOLD_MS = 8000;

let running = false;
let extDead = false;
let meetMode = "signer";
let framesSeen = 0;
let pending = false;
let rafId = 0;
const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { alpha: false, willReadFrequently: true });
let lastSpoken = "";
let prefs = { showLandmarks: false, outputMode: "subtitles" };
const hudState = {
  capturing: false,
  lastGloss: "—",
  spanish: "",
  statusText: "Buscando ILSA…",
};

function applyPrefsToPage() {
  postToPage({
    type: "LSA_PREFS",
    showSubtitles: meetMode === "hearing" || prefs.outputMode !== "audio",
    showLandmarks: Boolean(prefs.showLandmarks) && meetMode !== "hearing",
  });
}

function speakSpanish(text) {
  if (meetMode === "hearing") return;
  if (!text || prefs.outputMode === "subtitles") return;
  const clipped = String(text).trim();
  if (!clipped || clipped === lastSpoken) return;
  lastSpoken = clipped;
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(clipped);
    utter.lang = "es-AR";
    utter.rate = 1;
    window.speechSynthesis.speak(utter);
  } catch (_) {}
}

function stopVoice() {
  lastSpoken = "";
  try {
    window.speechSynthesis.cancel();
  } catch (_) {}
}

function extAlive() {
  try {
    return Boolean(!extDead && chrome.runtime && chrome.runtime.id);
  } catch (_) {
    return false;
  }
}

function onExtDead() {
  if (extDead) return;
  extDead = true;
  running = false;
  pending = false;
  cancelAnimationFrame(rafId);
  if (globalThis.__lsaMeetInterval) {
    clearInterval(globalThis.__lsaMeetInterval);
    globalThis.__lsaMeetInterval = 0;
  }
}

function sendExt(message, onDone) {
  if (!extAlive()) {
    onExtDead();
    if (onDone) onDone();
    return;
  }
  try {
    chrome.runtime.sendMessage(message, () => {
      try {
        void chrome.runtime.lastError;
      } catch (_) {
        onExtDead();
      }
      if (onDone) onDone();
    });
  } catch (_) {
    onExtDead();
    if (onDone) onDone();
  }
}

LsaPrefs.get()
  .then((p) => {
    prefs = p;
    applyPrefsToPage();
  })
  .catch(() => {});
try {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (!extAlive()) {
      onExtDead();
      return;
    }
    if (area !== "local" || !changes[LSA_PREFS_KEY]) return;
    prefs = normalizeLsaPrefs(changes[LSA_PREFS_KEY].newValue);
    applyPrefsToPage();
  });
} catch (_) {}

/**
 * Mensaje al mundo MAIN (inject-gum).
 * @param {object} payload Campos extra (`type`, `enabled`, `spanish`, …).
 */
function postToPage(payload) {
  window.postMessage({ source: "lsa-ext", ...payload }, "*");
}

/**
 * Crea el chip LSA arriba a la derecha si no existe.
 * @returns {HTMLElement}
 */
function ensureHud() {
  let root = document.getElementById("lsa-meet-root");
  if (root) return root;
  root = document.createElement("div");
  root.id = "lsa-meet-root";
  root.className = "";
  root.innerHTML = `
    <div class="lsa-card">
      <div class="lsa-row">
        <span class="lsa-badge">LSA</span>
        <span class="lsa-cap off" id="lsa-cap">OFF</span>
        <span class="lsa-status" id="lsa-status">Listo</span>
      </div>
      <div class="lsa-gloss" id="lsa-gloss">Última glosa: —</div>
      <div class="lsa-spanish" id="lsa-spanish"></div>
    </div>
  `;
  (document.body || document.documentElement).appendChild(root);
  return root;
}

/** Pinta `hudState` en el DOM. Siempre visible en Meet. */
function paintHud() {
  const root = ensureHud();
  root.classList.remove("lsa-hidden");
  const cap = root.querySelector("#lsa-cap");
  cap.textContent = hudState.capturing ? "ON" : "OFF";
  cap.className = "lsa-cap " + (hudState.capturing ? "on" : "off");
  root.querySelector("#lsa-status").textContent = hudState.statusText;
  root.querySelector("#lsa-gloss").textContent = "Última glosa: " + (hudState.lastGloss || "—");
  const sp = root.querySelector("#lsa-spanish");
  if (sp) sp.textContent = hudState.spanish || "";
}

/**
 * Fusiona un parche de estado (captions / debug) y refresca el HUD.
 * El español se borra solo a los {@link SUBTITLE_HOLD_MS} ms.
 * @param {object} partial
 */
function setHud(partial) {
  if (typeof partial.capturing === "boolean") hudState.capturing = partial.capturing;
  if (partial.lastGloss) hudState.lastGloss = partial.lastGloss;
  if (typeof partial.spanish === "string" && partial.spanish) {
    hudState.spanish = partial.spanish;
    if (window.__lsaHudCaptionTimer) clearTimeout(window.__lsaHudCaptionTimer);
    window.__lsaHudCaptionTimer = setTimeout(() => {
      hudState.spanish = "";
      paintHud();
    }, SUBTITLE_HOLD_MS);
  }
  if (partial.debug) {
    hudState.statusText = partial.debug;
  } else if (partial.status || partial.error) {
    const map = {
      escuchando: "Mostrá las manos para capturar",
      grabando: "Grabando seña",
      enviando: "Clasificando seña…",
      traduciendo: "Traduciendo…",
      listo: "Subtítulo en tu video",
      error: partial.error || "Error",
    };
    hudState.statusText = map[partial.status] || partial.error || partial.status;
  }
  paintHud();
}

/**
 * JPEG chico del video real (no del canvas de Meet) para Holistic.
 * @returns {{ width: number, height: number, dataUrl: string }|null}
 */
function grabJpeg() {
  const video = document.getElementById("lsa-real-cam");
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  const maxW = 320;
  let w = video.videoWidth;
  let h = video.videoHeight;
  if (w > maxW) {
    h = Math.round((h * maxW) / w);
    w = maxW;
  }
  if (grab.width !== w || grab.height !== h) {
    grab.width = w;
    grab.height = h;
  }
  grabCtx.drawImage(video, 0, 0, w, h);
  return { width: w, height: h, dataUrl: grab.toDataURL("image/jpeg", 0.62) };
}

let lastSendAt = 0;
/** Loop rAF: como máximo un JPEG cada ~70 ms, espera ACK implícito de `pending`. */
function loop() {
  if (!running || !extAlive()) return;
  const now = performance.now();
  if (!pending && now - lastSendAt >= 70) {
    const frame = grabJpeg();
    if (frame) {
      pending = true;
      lastSendAt = now;
      framesSeen += 1;
      sendExt({ type: "lsa-frame", ...frame }, () => {
        pending = false;
      });
    }
  }
  rafId = requestAnimationFrame(loop);
}

/** Activa el puente en Meet. `mode`: signer (LSA) o hearing (voz → subtítulos). */
function startBridge(mode) {
  const next = mode === "hearing" ? "hearing" : "signer";
  if (running && meetMode === next) return;
  if (running) stopBridge();
  meetMode = next;
  running = true;
  framesSeen = 0;
  pending = false;
  hudState.capturing = false;
  hudState.lastGloss = "—";
  ensureHud();
  postToPage({ type: "LSA_SET_ENABLED", enabled: true });
  applyPrefsToPage();
  cancelAnimationFrame(rafId);
  if (meetMode === "hearing") {
    setHud({ debug: "Hablá: el texto se pinta en tu cámara", capturing: false });
    postToPage({ type: "LSA_SPEECH_START" });
    return;
  }
  postToPage({ type: "LSA_SPEECH_STOP" });
  setHud({ status: "escuchando", capturing: false });
  sendExt({ type: "lsa-meet-offscreen-start", leftHanded: false });
  loop();
  setTimeout(() => {
    if (running && meetMode === "signer" && !framesSeen) {
      setHud({ debug: "Sin video. Apagá y prendé la cámara de Meet." });
    }
  }, 5000);
}

/** Apaga el gancho de gUM (Meet vuelve a la cámara nativa en el próximo pedido). */
function stopBridge() {
  running = false;
  pending = false;
  cancelAnimationFrame(rafId);
  stopVoice();
  postToPage({ type: "LSA_SPEECH_STOP" });
  postToPage({ type: "LSA_SET_ENABLED", enabled: false });
  setHud({ debug: "Encendé ILSA para traducir", capturing: false, lastGloss: "—" });
}

window.addEventListener("message", (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== "lsa-page") return;
  if (data.type === "LSA_PIPELINE" && running && !framesSeen) {
    setHud({ debug: "Mostrá las manos para capturar", capturing: false });
  }
  if (data.type === "LSA_SPEECH" && running && meetMode === "hearing") {
    const spanish = data.spanish;
    if (!spanish) return;
    setHud({ spanish, debug: "Transcribiendo…", capturing: true, lastGloss: "voz" });
    if (data.final) speakSpanish(spanish);
  }
  if (data.type === "LSA_SPEECH_ERROR") {
    setHud({ debug: "Voz: " + (data.error || "error"), capturing: false });
  }
});

function requestMeetSync() {
  sendExt({ type: "lsa-meet-sync" });
}

async function tickLocal() {
  let h = null;
  try {
    h = await LsaApi.health();
  } catch (_) {
    h = null;
  }
  if (!h || !h.ok) {
    if (running) stopBridge();
    else setHud({ debug: "Encendé ILSA para traducir", capturing: false });
    return;
  }
  const mode = h.mode === "hearing" ? "hearing" : "signer";
  startBridge(mode);
}

requestMeetSync();
tickLocal();
if (!globalThis.__lsaMeetInterval) {
  globalThis.__lsaMeetInterval = setInterval(() => {
    if (!extAlive()) {
      onExtDead();
      return;
    }
    requestMeetSync();
    tickLocal();
  }, 1500);
}

if (!globalThis.__lsaHudWatch) {
  globalThis.__lsaHudWatch = true;
  const watch = () => {
    if (!document.getElementById("lsa-meet-root")) {
      ensureHud();
      paintHud();
    }
  };
  const obs = new MutationObserver(watch);
  obs.observe(document.documentElement, { childList: true, subtree: true });
  setHud({ debug: "Buscando ILSA…", capturing: false });
}

try {
  chrome.runtime.onMessage.addListener((msg) => {
    if (!extAlive() || !msg || !msg.type) return;
    if (msg.type === "lsa-meet-content-start") startBridge(msg.mode);
    if (msg.type === "lsa-meet-content-stop") {
      if (running) stopBridge();
    }
    if (msg.type === "lsa-meet-content-error") {
      running = true;
      ensureHud();
      setHud({ debug: msg.error || "No se pudo conectar con ILSA", capturing: false });
    }
    if (msg.type === "lsa-landmarks") {
      postToPage({ type: "LSA_LANDMARKS", landmarks: msg.landmarks || null });
    }
    if (msg.type === "lsa-caption") {
      setHud(msg);
      postToPage({
        type: "LSA_CAPTION",
        spanish: msg.spanish,
        glosses: msg.glosses,
        status: msg.status,
      });
      if (msg.spanish) speakSpanish(msg.spanish);
    }
  });
} catch (_) {}
