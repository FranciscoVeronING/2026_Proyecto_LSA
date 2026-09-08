/**
 * Service worker MV3: única pieza que habla con popup, Meet y el documento offscreen.
 *
 * Mensajes:
 * - `lsa-meet-start-request` / `stop-request` — popup o botón del HUD.
 * - `lsa-frame` — JPEG desde meet.js → reenvío `lsa-offscreen-frame`.
 * - `lsa-caption` — estado desde offscreen → meet.js.
 */

const OFFSCREEN_URL = "offscreen.html";

let meetTabId = null;

/**
 * @returns {Promise<boolean>} true si ya hay un documento offscreen de esta extensión.
 */
async function hasOffscreen() {
  if (chrome.offscreen.hasDocument) {
    return chrome.offscreen.hasDocument();
  }
  const ctxs = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  return ctxs.length > 0;
}

/**
 * Crea `offscreen.html` (Holistic no puede correr en el service worker).
 * @returns {Promise<void>}
 */
async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["IFRAME_SCRIPTING", "BLOBS"],
    justification: "MediaPipe procesa el video local de Google Meet en un iframe aislado.",
  });
}

/**
 * Espera a que el documento offscreen exista (el grafo WASM tarda).
 * @param {number} [timeoutMs=8000]
 * @returns {Promise<void>}
 */
async function waitOffscreenReady(timeoutMs = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await hasOffscreen()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
}

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || !msg.type) return;

  if (msg.type === "lsa-meet-start-request") {
    startMeet(msg.tabId, msg.leftHanded)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }

  if (msg.type === "lsa-meet-stop-request") {
    stopMeet()
      .then(() => sendResponse({ ok: true }))
      .catch(() => sendResponse({ ok: true }));
    return true;
  }

  if (msg.type === "lsa-caption" && msg.tabId) {
    chrome.tabs.sendMessage(msg.tabId, msg).catch(() => {});
    return;
  }

  if (msg.type === "lsa-frame" && sender.tab) {
    meetTabId = sender.tab.id;
    chrome.runtime
      .sendMessage({
        type: "lsa-offscreen-frame",
        width: msg.width,
        height: msg.height,
        dataUrl: msg.dataUrl,
        buffer: msg.buffer,
      })
      .catch(() => {});
    sendResponse({ ok: true });
  }
});

/**
 * Manda un mensaje al offscreen; reintenta porque el SW puede despertar antes.
 * @param {object} message
 * @param {number} [tries=12]
 * @returns {Promise<any>}
 */
async function sendToOffscreen(message, tries = 12) {
  let lastErr = null;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await chrome.runtime.sendMessage(message);
      if (res) return res;
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw lastErr || new Error("El procesador de video no respondió.");
}

/**
 * Arranque Meet: offscreen + content script.
 * @param {number} tabId Pestaña `https://meet.google.com/…`.
 * @param {boolean} leftHanded
 * @returns {Promise<void>}
 */
async function startMeet(tabId, leftHanded) {
  const tab = await chrome.tabs.get(tabId);
  if (!tab.url || !tab.url.startsWith("https://meet.google.com/")) {
    throw new Error("Abrí una llamada de Google Meet y volvé a intentar.");
  }
  meetTabId = tabId;
  await ensureOffscreen();
  await waitOffscreenReady();
  const started = await sendToOffscreen({
    type: "lsa-meet-start",
    tabId,
    leftHanded: Boolean(leftHanded),
  });
  if (started && started.ok === false) {
    throw new Error(started.error || "No se pudo iniciar el intérprete.");
  }
  await chrome.tabs.sendMessage(tabId, { type: "lsa-meet-content-start" });
}

/** Para captura y HUD; no corta los tracks de la cámara de Meet. */
async function stopMeet() {
  const tabId = meetTabId;
  meetTabId = null;
  try {
    await chrome.runtime.sendMessage({ type: "lsa-meet-stop" });
  } catch (_) {}
  if (tabId) {
    try {
      await chrome.tabs.sendMessage(tabId, { type: "lsa-meet-content-stop" });
    } catch (_) {}
  }
}
