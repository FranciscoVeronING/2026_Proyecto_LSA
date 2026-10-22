/**
 * Cliente del motor en 127.0.0.1:8765.
 * Desde Meet (content script) Chrome trata el fetch como red local y lo
 * bloquea; ahí se pide al service worker. Popup/offscreen hablan directo.
 */

const API_BASE = "http://127.0.0.1:8765";

function fromMeetTab() {
  try {
    return location.protocol === "https:" && location.hostname === "meet.google.com";
  } catch (_) {
    return false;
  }
}

function parseBody(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function errorMessage(data, statusText) {
  const msg = (data && (data.detail || data.message)) || statusText;
  return typeof msg === "string" ? msg : JSON.stringify(msg);
}

async function apiViaWorker(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const reply = await chrome.runtime.sendMessage({
    type: "lsa-api",
    path,
    method,
    body: options.body || null,
  });
  if (!reply || !reply.ok) {
    throw new Error((reply && reply.error) || "ILSA no responde en 127.0.0.1:8765. Encendé el motor.");
  }
  return reply.data;
}

async function apiDirect(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      method,
      headers,
    });
  } catch (_) {
    throw new Error("Failed to fetch: ILSA no responde en 127.0.0.1:8765. Encendé el motor.");
  }
  const data = parseBody(await res.text());
  if (!res.ok) throw new Error(errorMessage(data, res.statusText));
  return data;
}

async function api(path, options = {}) {
  if (fromMeetTab()) return apiViaWorker(path, options);
  return apiDirect(path, options);
}

const LsaApi = {
  base: API_BASE,
  health: () => api("/health"),
  config: () => api("/config"),
  state: () => api("/state"),
  session: (leftHanded) =>
    api("/session", {
      method: "POST",
      body: JSON.stringify({ left_handed: leftHanded }),
    }),
  sign: (frames, client) =>
    api("/sign", {
      method: "POST",
      body: JSON.stringify({ frames, client: client || {} }),
    }),
  hearing: (spanish, final) =>
    api("/hearing", {
      method: "POST",
      body: JSON.stringify({ spanish, final: Boolean(final) }),
    }),
  stt: (wavB64, lang) =>
    api("/stt", {
      method: "POST",
      body: JSON.stringify({ wav_b64: wavB64, lang: lang || "es-ES" }),
    }),
  activity: () => api("/activity", { method: "POST", body: "{}" }),
  endUtterance: () => api("/utterance/end", { method: "POST", body: "{}" }),
  exeUrl: `${API_BASE}/download/exe`,
};
