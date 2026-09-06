const API_BASE = "http://127.0.0.1:8765";

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
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
  sign: (frames) =>
    api("/sign", {
      method: "POST",
      body: JSON.stringify({ frames }),
    }),
  activity: () => api("/activity", { method: "POST", body: "{}" }),
  endUtterance: () => api("/utterance/end", { method: "POST", body: "{}" }),
  clearConversation: () => api("/conversation/clear", { method: "POST", body: "{}" }),
  exeUrl: `${API_BASE}/download/exe`,
};
