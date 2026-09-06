const SUBTITLE_HOLD_MS = 8000;

let running = false;
let framesSeen = 0;
let pending = false;
let rafId = 0;
const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { alpha: false, willReadFrequently: true });
const hudState = {
  capturing: false,
  lastGloss: "—",
  spanish: "",
  statusText: "Mostrá las manos para capturar",
};

function postToPage(payload) {
  window.postMessage({ source: "lsa-ext", ...payload }, "*");
}

function ensureHud() {
  let root = document.getElementById("lsa-meet-root");
  if (root) return root;
  root = document.createElement("div");
  root.id = "lsa-meet-root";
  root.className = "lsa-hidden";
  root.innerHTML = `
    <div class="lsa-card">
      <div class="lsa-row">
        <span class="lsa-badge">LSA</span>
        <span class="lsa-cap off" id="lsa-cap">OFF</span>
        <span class="lsa-status" id="lsa-status">Listo</span>
        <button type="button" class="lsa-stop" id="lsa-stop">Detener</button>
      </div>
      <div class="lsa-gloss" id="lsa-gloss">Última glosa: —</div>
      <div class="lsa-spanish" id="lsa-spanish"></div>
    </div>
  `;
  (document.body || document.documentElement).appendChild(root);
  root.querySelector("#lsa-stop").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "lsa-meet-stop-request" });
  });
  return root;
}

function paintHud() {
  const root = ensureHud();
  if (!running) return;
  root.classList.remove("lsa-hidden");
  const cap = root.querySelector("#lsa-cap");
  cap.textContent = hudState.capturing ? "ON" : "OFF";
  cap.className = "lsa-cap " + (hudState.capturing ? "on" : "off");
  root.querySelector("#lsa-status").textContent = hudState.statusText;
  root.querySelector("#lsa-gloss").textContent = "Última glosa: " + (hudState.lastGloss || "—");
  const sp = root.querySelector("#lsa-spanish");
  if (sp) sp.textContent = hudState.spanish || "";
}

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

function hideHud() {
  const root = document.getElementById("lsa-meet-root");
  if (root) root.classList.add("lsa-hidden");
}

function grabJpeg() {
  const video = document.getElementById("lsa-real-cam");
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  const maxW = 192;
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
  return { width: w, height: h, dataUrl: grab.toDataURL("image/jpeg", 0.42) };
}

let lastSendAt = 0;
function loop() {
  if (!running) return;
  const now = performance.now();
  if (!pending && now - lastSendAt >= 70) {
    const frame = grabJpeg();
    if (frame) {
      pending = true;
      lastSendAt = now;
      framesSeen += 1;
      chrome.runtime.sendMessage({ type: "lsa-frame", ...frame }, () => {
        void chrome.runtime.lastError;
        pending = false;
      });
    }
  }
  rafId = requestAnimationFrame(loop);
}

function startBridge() {
  running = true;
  framesSeen = 0;
  pending = false;
  hudState.capturing = false;
  hudState.lastGloss = "—";
  ensureHud();
  postToPage({ type: "LSA_SET_ENABLED", enabled: true });
  setHud({ status: "escuchando", capturing: false });
  cancelAnimationFrame(rafId);
  loop();
  setTimeout(() => {
    if (running && !framesSeen) {
      setHud({ debug: "Sin video. Apagá y prendé la cámara de Meet." });
    }
  }, 5000);
}

function stopBridge() {
  running = false;
  pending = false;
  cancelAnimationFrame(rafId);
  postToPage({ type: "LSA_SET_ENABLED", enabled: false });
  hideHud();
}

window.addEventListener("message", (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== "lsa-page") return;
  if (data.type === "LSA_PIPELINE" && running && !framesSeen) {
    setHud({ debug: "Mostrá las manos para capturar", capturing: false });
  }
});

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || !msg.type) return;
  if (msg.type === "lsa-meet-content-start") startBridge();
  if (msg.type === "lsa-meet-content-stop") stopBridge();
  if (msg.type === "lsa-caption") {
    setHud(msg);
    postToPage({
      type: "LSA_CAPTION",
      spanish: msg.spanish,
      glosses: msg.glosses,
      status: msg.status,
    });
  }
});
