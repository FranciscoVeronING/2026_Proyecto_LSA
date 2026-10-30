/**
 * Isolated Meet world: HUD + JPEG to the SW (signer).
 * Hearing: Meet audio + chat → POST /stt and /hearing → glosses on the canvas.
 */

const SUBTITLE_HOLD_MS = 8000;

let running = false;
let extDead = false;
let meetMode = "signer";
let lastHealthOkAt = 0;
let ilsaLeftHanded = false;
let lastSentLeftHanded = null;
const ILSA_DOWN_MS = 8000;
let framesSeen = 0;
let pending = false;
let rafId = 0;
const grab = document.createElement("canvas");
const grabCtx = grab.getContext("2d", { alpha: false, willReadFrequently: true });
let lastSpoken = "";
let voiceReq = 0;
let lastVoiceNotice = "";
let lastHearingSent = { text: "", at: 0 };
let prefs = { showLandmarks: false, outputMode: "subtitles", hearingInput: "both" };
const hudState = {
  capturing: false,
  lastGloss: "—",
  glossKind: "",
  pendingText: "",
  spanish: "",
  statusText: "Buscando ILSA…",
  alert: false,
};

function hearingInputMode() {
  const value = prefs.hearingInput;
  return value === "voice" || value === "text" ? value : "both";
}

function wantsHearingVoice() {
  return hearingInputMode() !== "text";
}

function wantsHearingText() {
  return hearingInputMode() !== "voice";
}

function hearingHint() {
  const mode = hearingInputMode();
  if (mode === "voice") return "Hablá con el mic de Meet";
  if (mode === "text") return "Escribí en el chat de Meet";
  return "Hablá (mic de Meet) o escribí en el chat";
}

function applyPrefsToPage() {
  postToPage({
    type: "LSA_PREFS",
    showSubtitles: meetMode === "hearing" || prefs.outputMode !== "audio",
    showLandmarks: Boolean(prefs.showLandmarks) && meetMode !== "hearing",
    hearingInput: hearingInputMode(),
  });
}

function speakSpanish(text) {
  if (meetMode === "hearing") return;
  if (!text || prefs.outputMode === "subtitles") return;
  const clipped = String(text).trim();
  if (!clipped || clipped === lastSpoken) return;
  const id = ++voiceReq;
  let settled = false;
  postToPage({ type: "LSA_MIC_QUERY", id });
  const timer = setTimeout(() => finish(null), 800);
  function finish(open) {
    if (settled || id !== voiceReq) return;
    settled = true;
    clearTimeout(timer);
    window.removeEventListener("message", onReply);
    if (open === null) {
      setHud({ debug: "Voz: no se pudo confirmar el micrófono de Meet", capturing: false });
      return;
    }
    if (!open) {
      if (lastVoiceNotice !== clipped) {
        lastVoiceNotice = clipped;
        showVoiceMutedPopup();
      }
      return;
    }
    lastSpoken = clipped;
    LsaApi.tts(clipped)
      .then((res) => {
        const wav = res && res.wav_b64;
        if (!wav) throw new Error("sin audio");
        if (id !== voiceReq) return;
        postToPage({ type: "LSA_TTS_WAV", wavB64: wav, id });
      })
      .catch((err) => {
        if (lastSpoken === clipped) lastSpoken = "";
        setHud({ debug: "Voz: " + ((err && err.message) || "no se pudo leer"), capturing: false });
      });
  }
  function onReply(event) {
    const data = event.data;
    if (!data || data.source !== "lsa-page" || data.type !== "LSA_MIC_STATE" || data.id !== id) return;
    finish(Boolean(data.open));
  }
  window.addEventListener("message", onReply);
}

function showVoiceMutedPopup() {
  let pop = document.getElementById("lsa-voice-pop");
  if (!pop) {
    pop = document.createElement("div");
    pop.id = "lsa-voice-pop";
    pop.innerHTML =
      '<div class="lsa-voice-card" role="dialog" aria-modal="true" aria-labelledby="lsa-voice-title">' +
      '<p id="lsa-voice-title">El micrófono de Meet está silenciado. Con el micrófono en silencio no se puede reproducir la voz en off para el resto de la llamada.</p>' +
      '<button type="button" id="lsa-voice-ok">Entendido</button>' +
      "</div>";
    document.documentElement.appendChild(pop);
    pop.querySelector("#lsa-voice-ok").addEventListener("click", () => {
      pop.hidden = true;
    });
  }
  pop.hidden = false;
}

function stopVoice() {
  lastSpoken = "";
  voiceReq += 1;
  postToPage({ type: "LSA_TTS_STOP" });
  try {
    window.speechSynthesis.cancel();
  } catch (_) {}
}

function transcribeMeetAudio(wavB64) {
  if (!running || meetMode !== "hearing" || !wantsHearingVoice() || !wavB64) return;
  setHud({ debug: "Transcribiendo tu voz…", capturing: true, lastGloss: "voice" });
  LsaApi.stt(wavB64, "es-ES")
    .then((s) => {
      const spanish = String((s && s.spanish) || "").trim();
      if (!spanish) {
        setHud({
          debug: wantsHearingText()
            ? "No se entendió el audio; repetí o usá el chat"
            : "No se entendió el audio; repetí",
          capturing: false,
        });
        return;
      }
      sendToHearing(spanish, "voice");
    })
    .catch((err) => {
      setHud({
        debug: "STT: " + ((err && err.message) || "error"),
        capturing: false,
      });
    });
}

function applyAvatarClip(clip) {
  if (!clip || clip.ok === false || !(clip.frames && clip.frames.length)) {
    setHud({ debug: "Esqueleto en espera: " + ((clip && clip.error) || "sin clips") });
    return;
  }
  postToPage({ type: "LSA_AVATAR", clip });
  const miss = (clip.missing || []).filter(Boolean);
  setHud({
    debug: miss.length ? "Sin clip: " + miss.join(", ") : "Esqueleto en el video",
  });
}

/** Spanish (voice or chat) → POST /hearing → glosses burned into the video. */
function sendToHearing(spanish, source) {
  const text = String(spanish || "").trim();
  if (!text || !running || meetMode !== "hearing") return;
  const now = Date.now();
  if (text === lastHearingSent.text && now - lastHearingSent.at < 2500) return;
  lastHearingSent = { text, at: now };
  setHud({ debug: "Traduciendo a glosas…", capturing: true, lastGloss: source });
  LsaApi.hearing(text, true)
    .then((s) => {
      if (s && s.semantic_ready === false) {
        setHud({
          debug: "El traductor no responde. Reiniciá ILSA.",
          capturing: false,
          alert: true,
        });
        return;
      }
      const glosses = String((s && (s.caption || s.glosses)) || "").trim();
      if (!glosses) {
        setHud({ debug: "El traductor no devolvió glosas", capturing: false });
        return;
      }
      const tPresent = performance.now();
      setHud({
        spanish: glosses,
        debug: "Glosas en tu video",
        capturing: false,
        lastGloss: source,
      });
      postToPage({
        type: "LSA_CAPTION",
        line: glosses,
        spanish: glosses,
        glosses,
      });
      applyAvatarClip(s.avatar);
      const meet_ms = performance.now() - tPresent;
      LsaApi.clientMetrics({ channel: "inverse", delivery_ms: meet_ms }).catch(() => {});
    })
    .catch((err) => {
      setHud({
        debug: "El traductor no responde. Reiniciá ILSA.",
        capturing: false,
        alert: true,
      });
    });
}

function chatBoxText(el) {
  if (!el) return "";
  if (typeof el.value === "string" && el.value.trim()) return el.value.trim();
  const dt = el.getAttribute && el.getAttribute("data-initial-value");
  if (dt && dt.trim()) return dt.trim();
  return String(el.innerText || el.textContent || "").trim();
}

function chatRoot(el) {
  if (!el || !el.closest) return null;
  return el.closest("textarea, input[type='text'], [contenteditable='true'], [role='textbox']");
}

function isMeetChatBox(el) {
  const root = chatRoot(el);
  if (!root) return false;
  if (root.closest && root.closest("#lsa-meet-root")) return false;
  return true;
}

function hookMeetChat() {
  if (window.__lsaChatHookedV2) return;
  window.__lsaChatHookedV2 = true;
  document.addEventListener(
    "keydown",
    (ev) => {
      if (!running || meetMode !== "hearing" || !wantsHearingText()) return;
      if (ev.key !== "Enter" || ev.shiftKey) return;
      const root = chatRoot(ev.target);
      if (!root || !isMeetChatBox(root)) return;
      const text = chatBoxText(root);
      if (text) sendToHearing(text, "chat");
    },
    true
  );
  document.addEventListener(
    "click",
    (ev) => {
      if (!running || meetMode !== "hearing" || !wantsHearingText()) return;
      const btn = ev.target && ev.target.closest && ev.target.closest("button");
      if (!btn) return;
      const lab = ((btn.getAttribute("aria-label") || "") + " " + (btn.textContent || "")).toLowerCase();
      if (!/enviar|send message|send/.test(lab)) return;
      const root =
        chatRoot(document.activeElement) ||
        document.querySelector("[role='textbox'], textarea, [contenteditable='true']");
      const text = chatBoxText(root);
      if (text) sendToHearing(text, "chat");
    },
    true
  );
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
    if (running && meetMode === "hearing") {
      setHud({ debug: hearingHint(), capturing: false });
    }
  });
} catch (_) {}

/**
 * Message to the MAIN world (camera-hook).
 * @param {object} payload Extra fields (`type`, `enabled`, `spanish`, …).
 */
function postToPage(payload) {
  window.postMessage({ source: "lsa-ext", ...payload }, "*");
}

/**
 * Create the LSA chip at the top-right if it does not exist.
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
      <div class="lsa-pending" id="lsa-pending" hidden></div>
      <div class="lsa-hear" id="lsa-hear" hidden>
        <span class="lsa-hear-label">Captura</span>
        <button type="button" class="lsa-hear-btn" data-hear="voice">Voz</button>
        <button type="button" class="lsa-hear-btn" data-hear="text">Texto</button>
        <button type="button" class="lsa-hear-btn" data-hear="both">Ambos</button>
      </div>
      <div class="lsa-spanish" id="lsa-spanish"></div>
    </div>
  `;
  (document.body || document.documentElement).appendChild(root);
  root.querySelectorAll("[data-hear]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const hearingInput = btn.getAttribute("data-hear");
      if (hearingInput !== "voice" && hearingInput !== "text" && hearingInput !== "both") return;
      LsaPrefs.set({ hearingInput });
    });
  });
  bindHudDrag(root);
  LsaPrefs.get()
    .then((p) => {
      if (p.hudLeft == null || p.hudTop == null) return;
      placeHud(root, p.hudLeft, p.hudTop);
    })
    .catch(() => {});
  return root;
}

/**
 * @param {HTMLElement} root
 * @param {number} left
 * @param {number} top
 */
function placeHud(root, left, top) {
  const card = root.querySelector(".lsa-card");
  const width = (card && card.offsetWidth) || 280;
  const height = (card && card.offsetHeight) || 72;
  const x = Math.min(Math.max(8, left), Math.max(8, window.innerWidth - width - 8));
  const y = Math.min(Math.max(8, top), Math.max(8, window.innerHeight - height - 8));
  root.style.left = x + "px";
  root.style.top = y + "px";
  root.style.right = "auto";
}

/**
 * Drag the chip from anywhere that is not a button.
 * @param {HTMLElement} root
 */
function bindHudDrag(root) {
  if (root.dataset.lsaDrag === "1") return;
  root.dataset.lsaDrag = "1";
  const card = root.querySelector(".lsa-card");
  if (!card) return;
  let drag = null;
  card.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const target = ev.target;
    if (target && target.closest && target.closest("button, a, input, textarea, select")) return;
    const rect = root.getBoundingClientRect();
    drag = { dx: ev.clientX - rect.left, dy: ev.clientY - rect.top };
    card.classList.add("lsa-dragging");
    if (card.setPointerCapture) card.setPointerCapture(ev.pointerId);
  });
  card.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    placeHud(root, ev.clientX - drag.dx, ev.clientY - drag.dy);
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    card.classList.remove("lsa-dragging");
    const rect = root.getBoundingClientRect();
    if (extAlive()) {
      LsaPrefs.set({ hudLeft: Math.round(rect.left), hudTop: Math.round(rect.top) }).catch(() => {});
    }
  };
  card.addEventListener("pointerup", end);
  card.addEventListener("pointercancel", end);
  window.addEventListener("resize", () => {
    if (root.style.left === "") return;
    const rect = root.getBoundingClientRect();
    placeHud(root, rect.left, rect.top);
  });
}

/** Paint `hudState` into the DOM. Always visible in Meet. */
function paintHud() {
  const root = ensureHud();
  root.classList.remove("lsa-hidden");
  const card = root.querySelector(".lsa-card");
  if (card) card.classList.toggle("lsa-alert", Boolean(hudState.alert));
  const cap = root.querySelector("#lsa-cap");
  cap.textContent = hudState.capturing ? "ON" : "OFF";
  cap.className = "lsa-cap " + (hudState.capturing ? "on" : "off");
  root.querySelector("#lsa-status").textContent = hudState.statusText;
  const glossEl = root.querySelector("#lsa-gloss");
  let token = hudState.lastGloss || "—";
  if (token === "voice") token = "voz";
  else if (token === "chat" || token === "text") token = "chat";
  let glossLine = "Última glosa: " + token;
  if (token && token !== "—" && hudState.glossKind === "added") {
    glossLine = "En el mensaje: " + token;
  } else if (token && token !== "—" && hudState.glossKind === "confidence") {
    glossLine = "Poca confianza: " + token;
  } else if (token && token !== "—" && hudState.glossKind === "repeat") {
    glossLine = "No se agregó: " + token;
  }
  glossEl.textContent = glossLine;
  glossEl.className = "lsa-gloss" + (hudState.glossKind ? " lsa-gloss-" + hudState.glossKind : "");
  const pendingEl = root.querySelector("#lsa-pending");
  if (pendingEl) {
    const pending = String(hudState.pendingText || "").trim();
    pendingEl.hidden = !pending;
    pendingEl.textContent = pending ? "Enunciado: " + pending : "";
  }
  const sp = root.querySelector("#lsa-spanish");
  if (sp) sp.textContent = hudState.spanish || "";
  const hear = root.querySelector("#lsa-hear");
  if (hear) hear.hidden = meetMode !== "hearing";
  root.querySelectorAll("[data-hear]").forEach((btn) => {
    btn.classList.toggle("on", btn.getAttribute("data-hear") === hearingInputMode());
  });
}

/**
 * Merge a state patch (captions / debug) and refresh the HUD.
 * Spanish clears on its own after {@link SUBTITLE_HOLD_MS} ms.
 * @param {object} partial
 */
function setHud(partial) {
  if (typeof partial.alert === "boolean") hudState.alert = partial.alert;
  if (typeof partial.capturing === "boolean") hudState.capturing = partial.capturing;
  if (partial.lastGloss) {
    hudState.lastGloss = partial.lastGloss;
    if (partial.lastGloss === "—") {
      hudState.glossKind = "";
      hudState.pendingText = "";
    }
  }
  if (typeof partial.glossAdded === "boolean") {
    if (partial.glossAdded) hudState.glossKind = "added";
    else if (partial.glossReject === "repeat") hudState.glossKind = "repeat";
    else hudState.glossKind = "confidence";
  }
  if (typeof partial.pendingText === "string") hudState.pendingText = partial.pendingText;
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
  if (hudState.alert) {
    hudState.statusText = "El traductor no responde. Reiniciá ILSA.";
  }
  paintHud();
}

/**
 * Small JPEG of the real video (not Meet's canvas) for Holistic.
 * @returns {{ width: number, height: number, dataUrl: string }|null}
 */
function grabJpeg() {
  const video = document.getElementById("lsa-real-cam");
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  const maxW = 240;
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
  return { width: w, height: h, dataUrl: grab.toDataURL("image/jpeg", 0.5) };
}

let lastSendAt = 0;
let meetCamOpen = true;
/** rAF loop: at most one JPEG every ~70 ms; waits for implicit ACK via `pending`. */
function loop() {
  if (!running || !extAlive()) return;
  const now = performance.now();
  if (meetMode === "signer" && meetCamOpen && !pending && now - lastSendAt >= 70) {
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

/** Enable the Meet bridge. `mode`: signer (LSA) or hearing (voice → captions). */
function startBridge(mode) {
  const next = mode === "hearing" ? "hearing" : "signer";
  if (running && meetMode === next) {
    if (next === "signer" && lastSentLeftHanded !== ilsaLeftHanded) {
      lastSentLeftHanded = ilsaLeftHanded;
      sendExt({ type: "lsa-meet-offscreen-start", leftHanded: ilsaLeftHanded });
    }
    return;
  }
  if (running) stopBridge();
  meetMode = next;
  running = true;
  framesSeen = 0;
  pending = false;
  hudState.capturing = false;
  hudState.lastGloss = "—";
  hudState.glossKind = "";
  hudState.pendingText = "";
  ensureHud();
  postToPage({ type: "LSA_SET_ENABLED", enabled: true, hearing: meetMode === "hearing" });
  applyPrefsToPage();
  cancelAnimationFrame(rafId);
  if (meetMode === "hearing") {
    setHud({
      debug: hearingHint(),
      capturing: false,
    });
    hookMeetChat();
    return;
  }
  postToPage({ type: "LSA_SPEECH_STOP" });
  setHud({ status: "escuchando", capturing: false });
  lastSentLeftHanded = ilsaLeftHanded;
  sendExt({ type: "lsa-meet-offscreen-start", leftHanded: ilsaLeftHanded });
  loop();
  setTimeout(() => {
    if (running && meetMode === "signer" && !framesSeen) {
      setHud({ debug: "Sin video. Apagá y prendé la cámara de Meet." });
    }
  }, 5000);
}

/** Disable the gUM hook (Meet returns to the native camera on the next request). */
function stopBridge() {
  running = false;
  lastSentLeftHanded = null;
  pending = false;
  cancelAnimationFrame(rafId);
  stopVoice();
  postToPage({ type: "LSA_SPEECH_STOP" });
  postToPage({ type: "LSA_SET_ENABLED", enabled: false });
  setHud({ debug: "Encendé ILSA para traducir", capturing: false, lastGloss: "—", alert: false });
}

window.addEventListener("message", (event) => {
  if (event.source !== window) return;
  const data = event.data;
  if (!data || data.source !== "lsa-page") return;
  if (data.type === "LSA_CAMERA") {
    const open = data.open !== false;
    if (open !== meetCamOpen) {
      meetCamOpen = open;
      if (!open) {
        sendExt({ type: "lsa-capture-reset" });
        if (running && meetMode === "signer") {
          setHud({ debug: "Cámara de Meet apagada", capturing: false });
        }
      } else if (running && meetMode === "signer") {
        setHud({ status: "escuchando", capturing: false });
      }
    }
  }
  if (data.type === "LSA_PIPELINE" && running && meetMode !== "hearing" && !framesSeen) {
    setHud({ debug: "Mostrá las manos para capturar", capturing: false });
  }
  if (data.type === "LSA_AUDIO_WAV" && running && meetMode === "hearing" && wantsHearingVoice()) {
    transcribeMeetAudio(data.wavB64);
  }
  if (data.type === "LSA_SPEECH_UI" && meetMode === "hearing" && wantsHearingVoice() && data.debug) {
    setHud({ debug: data.debug, capturing: false });
  }
  if (data.type === "LSA_VOICE_BLOCKED") {
    showVoiceMutedPopup();
    return;
  }
  if (data.type === "LSA_SPEECH_ERROR") {
    if (meetMode === "hearing" && !wantsHearingVoice()) return;
    setHud({ debug: "Voz: " + (data.error || "error"), capturing: false });
  }
  if (data.type === "LSA_FPS" && typeof data.fps === "number") {
    const now = Date.now();
    if (!globalThis.__lsaLastFpsPost || now - globalThis.__lsaLastFpsPost >= 5000) {
      globalThis.__lsaLastFpsPost = now;
      LsaApi.clientMetrics({
        channel: meetMode === "hearing" ? "inverse" : "direct",
        fps: data.fps,
      }).catch(() => {});
    }
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
  if (h && h.ok) {
    lastHealthOkAt = Date.now();
    ilsaLeftHanded = Boolean(h.left_handed);
    startBridge(h.mode === "hearing" ? "hearing" : "signer");
    if (h.semantic_ready === false) {
      setHud({
        debug: "El traductor no responde. Reiniciá ILSA.",
        capturing: false,
        alert: true,
      });
    } else if (hudState.alert) {
      setHud({
        alert: false,
        debug: meetMode === "hearing" ? hearingHint() : "Mostrá las manos para capturar",
      });
    }
    return;
  }
  if (running && lastHealthOkAt && Date.now() - lastHealthOkAt >= ILSA_DOWN_MS) {
    stopBridge();
    return;
  }
  if (!running) setHud({ debug: "Encendé ILSA para traducir", capturing: false });
}

requestMeetSync();
tickLocal();
if (!globalThis.__lsaMeetInterval) {
  globalThis.__lsaMeetInterval = setInterval(() => {
    if (!extAlive()) {
      onExtDead();
      return;
    }
    tickLocal();
  }, 2000);
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
      if (meetMode === "hearing") return;
      const t0 = performance.now();
      setHud(msg);
      const spanish = typeof msg.spanish === "string" ? msg.spanish.trim() : "";
      if (spanish) {
        postToPage({ type: "LSA_CAPTION", line: spanish });
        speakSpanish(spanish);
        const delivery_ms = performance.now() - t0;
        LsaApi.clientMetrics({ channel: "direct", delivery_ms }).catch(() => {});
      }
    }
  });
} catch (_) {}
