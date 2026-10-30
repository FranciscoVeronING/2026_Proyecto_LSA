/**
 * Engine client on 127.0.0.1:8765.
 * From Meet (content script) Chrome treats the fetch as a local network
 * request and blocks it; then we ask the service worker. Popup/offscreen talk directly.
 */

const API_BASE = "http://127.0.0.1:8765";

function isExtensionPage() {
  try {
    return location.protocol === "chrome-extension:";
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
  if (isExtensionPage()) return apiDirect(path, options);
  return apiViaWorker(path, options);
}

function chromeProcessProbe() {
  const out = {};
  try {
    const mem = performance.memory;
    if (mem && typeof mem.usedJSHeapSize === "number") {
      out.js_heap_mb = Math.round((mem.usedJSHeapSize / 1048576) * 10) / 10;
    }
    if (mem && typeof mem.totalJSHeapSize === "number") {
      out.js_heap_total_mb = Math.round((mem.totalJSHeapSize / 1048576) * 10) / 10;
    }
    if (mem && typeof mem.jsHeapSizeLimit === "number") {
      out.js_heap_limit_mb = Math.round((mem.jsHeapSizeLimit / 1048576) * 10) / 10;
    }
  } catch (_) {}
  return out;
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
  sign: (frames, client) => {
    const merged = { ...(client || {}) };
    const body = JSON.stringify({ frames, client: merged });
    merged.payload_kb = Math.round((body.length / 1024) * 10) / 10;
    return api("/sign", {
      method: "POST",
      body: JSON.stringify({ frames, client: merged }),
    });
  },
  hearing: (spanish, final, extra) =>
    api("/hearing", {
      method: "POST",
      body: JSON.stringify({
        spanish,
        final: Boolean(final),
        stt_ms: extra && extra.stt_ms != null ? extra.stt_ms : undefined,
      }),
    }),
  stt: (wavB64, lang) =>
    api("/stt", {
      method: "POST",
      body: JSON.stringify({ wav_b64: wavB64, lang: lang || "es-ES" }),
    }),
  tts: (text) =>
    api("/tts", {
      method: "POST",
      body: JSON.stringify({ text: text || "" }),
    }),
  activity: () => api("/activity", { method: "POST", body: "{}" }),
  endUtterance: (payload) =>
    api("/utterance/end", {
      method: "POST",
      body: JSON.stringify(payload || {}),
    }),
  clientMetrics: (payload) =>
    api("/metrics/client", {
      method: "POST",
      body: JSON.stringify({ ...chromeProcessProbe(), ...(payload || {}) }),
    }),
  exeUrl: `${API_BASE}/download/exe`,
};
