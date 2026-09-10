/**
 * Service worker MV3: única pieza que habla con popup, Meet y el documento offscreen.
 *
 * Mensajes:
 * - `lsa-meet-sync` — Meet pregunta si ILSA está encendido; arranca o corta solo.
 * - `lsa-meet-start-request` / `stop-request` — compatibilidad / apagado.
 * - `lsa-frame` — JPEG desde meet.js → reenvío `lsa-offscreen-frame`.
 * - `lsa-caption` — estado desde offscreen → meet.js.
 */

importScripts("lib/prefs.js");

const OFFSCREEN_URL = "offscreen.html";

let meetTabId = null;
let meetModeOn = null;
let offscreenPort = null;
const offscreenAcks = new Map();

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

chrome.runtime.onConnect.addListener((port) => {
  if (!port || port.name !== "lsa-offscreen") return;
  offscreenPort = port;
  port.onMessage.addListener((msg) => {
    if (!msg || msg.type !== "lsa-ack" || !msg.id) return;
    const waiter = offscreenAcks.get(msg.id);
    if (!waiter) return;
    offscreenAcks.delete(msg.id);
    waiter.resolve(msg);
  });
  port.onDisconnect.addListener(() => {
    if (offscreenPort === port) offscreenPort = null;
  });
  LsaPrefs.get().then((prefs) => {
    try {
      port.postMessage({ type: "lsa-prefs", prefs });
    } catch (_) {}
  });
});

LsaPrefs.onChange((prefs) => {
  if (!offscreenPort) return;
  try {
    offscreenPort.postMessage({ type: "lsa-prefs", prefs });
  } catch (_) {}
});

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
  }
  wakeMeetTabs();
});

chrome.runtime.onStartup.addListener(() => {
  wakeMeetTabs();
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || !msg.type) return;

  if (msg.type === "lsa-meet-sync") {
    const tabId = (sender.tab && sender.tab.id) || msg.tabId;
    if (tabId) {
      syncMeetTab(tabId)
        .then(() => sendResponse({ ok: true, on: Boolean(meetTabId) }))
        .catch(() => sendResponse({ ok: true, on: false }));
      return true;
    }
  }

  if (msg.type === "lsa-meet-offscreen-start") {
    const tabId = (sender.tab && sender.tab.id) || msg.tabId;
    if (!tabId) return;
    startMeet(tabId, msg.leftHanded)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message || String(err) }));
    return true;
  }

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

  if (msg.type === "lsa-landmarks" && msg.tabId) {
    chrome.tabs.sendMessage(msg.tabId, msg).catch(() => {});
    return;
  }

  if (msg.type === "lsa-frame" && sender.tab) {
    meetTabId = sender.tab.id;
    const frame = {
      type: "lsa-offscreen-frame",
      width: msg.width,
      height: msg.height,
      dataUrl: msg.dataUrl,
      buffer: msg.buffer,
    };
    if (offscreenPort) {
      try {
        offscreenPort.postMessage(frame);
      } catch (_) {}
    }
    sendResponse({ ok: true });
  }
});

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  const url = tab && tab.url;
  if (url && !url.startsWith("https://meet.google.com/") && tabId === meetTabId) {
    stopMeet().catch(() => {});
    return;
  }
  if (info.status === "complete" && url && url.startsWith("https://meet.google.com/")) {
    syncMeetTab(tabId).catch(() => {});
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === meetTabId) stopMeet().catch(() => {});
});

if (chrome.webNavigation && chrome.webNavigation.onCompleted) {
  const onMeetNav = (details) => {
    if (details.frameId !== 0) return;
    const url = details.url || "";
    if (!url.startsWith("https://meet.google.com/")) return;
    syncMeetTab(details.tabId).catch(() => {});
  };
  chrome.webNavigation.onCompleted.addListener(onMeetNav);
  chrome.webNavigation.onHistoryStateUpdated.addListener(onMeetNav);
}

chrome.tabs.onActivated.addListener((info) => {
  chrome.tabs.get(info.tabId).then((tab) => {
    if (tab.url && tab.url.startsWith("https://meet.google.com/")) {
      syncMeetTab(tab.id).catch(() => {});
    }
  }).catch(() => {});
});

/**
 * @returns {Promise<{ok: boolean, mode?: string}|null>}
 */
async function pingHealth() {
  const urls = ["http://127.0.0.1:8765/health", "http://localhost:8765/health"];
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const h = await res.json();
      if (h && h.ok) return h;
    } catch (_) {}
  }
  return null;
}

/**
 * Si ILSA está encendido, traduce en Meet. Si se apagó el exe, corta.
 * @param {number} tabId
 */
async function syncMeetTab(tabId) {
  let tab;
  try {
    tab = await chrome.tabs.get(tabId);
  } catch {
    if (tabId === meetTabId) await stopMeet();
    return;
  }
  const url = tab.url || tab.pendingUrl || "";
  if (!url.startsWith("https://meet.google.com/")) {
    if (tabId === meetTabId) await stopMeet();
    return;
  }
  const h = await pingHealth();
  if (!h) {
    if (meetTabId) {
      try {
        if (offscreenPort) offscreenPort.postMessage({ type: "lsa-meet-stop" });
      } catch (_) {}
      meetTabId = null;
      meetModeOn = null;
    }
    return;
  }
  const mode = h.mode === "hearing" ? "hearing" : "signer";
  if (meetTabId === tabId && meetModeOn === mode) return;
  try {
    await startMeet(tabId, false);
  } catch (err) {
    const message = (err && err.message) || String(err);
    chrome.tabs
      .sendMessage(tabId, { type: "lsa-meet-content-error", error: message })
      .catch(() => {});
    throw err;
  }
}

async function injectMeetScripts(tabId) {
  if (!chrome.scripting || !chrome.scripting.executeScript) return;
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["lib/api.js", "lib/prefs.js", "content/meet.js"],
      world: "ISOLATED",
    });
  } catch (_) {}
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content/inject-gum.js"],
      world: "MAIN",
    });
  } catch (_) {}
}

async function wakeMeetTabs() {
  const tabs = await chrome.tabs.query({ url: "https://meet.google.com/*" });
  for (const tab of tabs) {
    if (!tab.id) continue;
    await injectMeetScripts(tab.id);
    syncMeetTab(tab.id).catch(() => {});
  }
}

/**
 * Manda un mensaje al offscreen; reintenta porque el SW puede despertar antes.
 * @param {object} message
 * @param {number} [tries=12]
 * @returns {Promise<any>}
 */
async function sendToOffscreen(message, tries = 40) {
  await ensureOffscreen();
  await waitOffscreenReady();
  let lastErr = null;
  for (let i = 0; i < tries; i++) {
    if (offscreenPort) {
      const id = String(Date.now()) + "-" + i;
      const ack = new Promise((resolve, reject) => {
        const t = setTimeout(() => {
          offscreenAcks.delete(id);
          reject(new Error("El procesador de video no respondió."));
        }, 25000);
        offscreenAcks.set(id, {
          resolve: (msg) => {
            clearTimeout(t);
            resolve(msg);
          },
        });
      });
      try {
        offscreenPort.postMessage({ ...message, id });
        return await ack;
      } catch (err) {
        lastErr = err;
        offscreenAcks.delete(id);
      }
    }
    await new Promise((r) => setTimeout(r, 80));
  }
  throw lastErr || new Error("El procesador de video no conectó.");
}

/**
 * Arranque Meet: offscreen + content script.
 * @param {number} tabId Pestaña `https://meet.google.com/…`.
 * @param {boolean} leftHanded
 * @returns {Promise<void>}
 */
async function startMeet(tabId, leftHanded) {
  const tab = await chrome.tabs.get(tabId);
  const url = tab.url || tab.pendingUrl || "";
  if (!url.startsWith("https://meet.google.com/")) {
    throw new Error("Abrí una llamada de Google Meet y volvé a intentar.");
  }
  const h = await pingHealth();
  if (!h) throw new Error("ILSA no está encendido. Elegí un modo y dale a Encender.");
  const mode = h.mode === "hearing" ? "hearing" : "signer";
  if (meetTabId === tabId && meetModeOn === mode) return;
  if (meetTabId && (meetTabId !== tabId || meetModeOn !== mode)) {
    await stopMeet();
  }
  try {
    await chrome.tabs.sendMessage(tabId, { type: "lsa-meet-content-start", mode });
  } catch (_) {
    await injectMeetScripts(tabId);
    await chrome.tabs.sendMessage(tabId, { type: "lsa-meet-content-start", mode });
  }
  if (mode === "signer") {
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
  }
  meetTabId = tabId;
  meetModeOn = mode;
}

/** Para captura y HUD; no corta los tracks de la cámara de Meet. */
async function stopMeet() {
  const tabId = meetTabId;
  meetTabId = null;
  meetModeOn = null;
  try {
    if (offscreenPort) offscreenPort.postMessage({ type: "lsa-meet-stop" });
  } catch (_) {}
  try {
    await chrome.runtime.sendMessage({ type: "lsa-meet-stop" });
  } catch (_) {}
  if (tabId) {
    try {
      await chrome.tabs.sendMessage(tabId, { type: "lsa-meet-content-stop" });
    } catch (_) {}
  }
}
