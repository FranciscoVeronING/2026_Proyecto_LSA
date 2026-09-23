/**
 * Cliente del motor en 127.0.0.1:8765.
 * `sign` manda puntos 3D, no video — Holistic corre en sandbox.js.
 */

/** Origen fijado en el manifest (host_permissions). */
const API_BASE = "http://127.0.0.1:8765";

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (err) {
    throw new Error(
      "Failed to fetch: ILSA no responde en 127.0.0.1:8765. Encendé el motor."
    );
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
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
  activity: () => api("/activity", { method: "POST", body: "{}" }),
  endUtterance: () => api("/utterance/end", { method: "POST", body: "{}" }),
  clearConversation: () => api("/conversation/clear", { method: "POST", body: "{}" }),
  exeUrl: `${API_BASE}/download/exe`,
};
