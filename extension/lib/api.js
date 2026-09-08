/**
 * Cliente HTTP del motor local (FastAPI en 127.0.0.1:8765).
 *
 * `LsaApi` es el catálogo completo de rutas que usa la extensión. No hay
 * otras llamadas al backend desde JS.
 *
 * Importante: `sign` manda landmarks (puntos 3D), no video. MediaPipe corre
 * en el navegador (`sandbox.js`). El servidor clasifica; no extrae esqueleto.
 */

/** Origen fijo del backend. Chrome MV3 solo permite este host si está en el manifest. */
const API_BASE = "http://127.0.0.1:8765";

/**
 * Funcion para hacer GET/POST genérico contra el backend.
 *
 * @param {string} path Ruta absoluta en el servidor, p. ej. `"/health"`.
 * @param {RequestInit} [options] Opciones de `fetch` (method, body, headers).
 * @returns {Promise<any>} Cuerpo JSON parseado.
 * @throws {Error} Si `res.ok` es falso.
 */
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

/**
 * Única lista de endpoints. Cada método = una ruta de `src/backend/server.py`.
 *
 * @property {string} base Origen `http://127.0.0.1:8765`.
 * @property {() => Promise<object>} health ¿Pipeline, clasificador y LLM listos?
 * @property {() => Promise<object>} config Umbrales de captura (`classifier.config`).
 * @property {() => Promise<object>} state Snapshot de glosas / español.
 * @property {(leftHanded: boolean) => Promise<object>} session Reinicia buffer y mano dominante.
 * @property {(frames: object[]) => Promise<object>} sign Clasifica una seña ya recortada.
 *   `frames` = `{pose, left_hand, right_hand}[]` (Holistic), no JPEG.
 * @property {() => Promise<{ok: boolean}>} activity “Sigo señando”: retrasa el cierre.
 * @property {() => Promise<object>} endUtterance Cierra glosas y pide español.
 * @property {() => Promise<object>} clearConversation Vacía memoria de la LLM.
 * @property {string} exeUrl Descarga del .exe/.bat si el backend lo sirve.
 */
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
