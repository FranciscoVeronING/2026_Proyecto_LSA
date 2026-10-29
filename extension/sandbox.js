/**
 * Sandbox iframe: **landmarks are extracted here** (MediaPipe Holistic WASM).
 *
 * The parent sends pixels `{ type: "frame", width, height, buffer }`.
 * Holistic returns a skeleton; this script posts
 * `{ type: "landmarks", pose, left_hand, right_hand }` to offscreen.
 * The backend never sees the JPEG.
 */

/**
 * @param {Array<{x:number,y:number,z?:number,visibility?:number}>|null|undefined} lms
 * @returns {object[]|null}
 */
function packLandmarks(lms) {
  if (!lms || !lms.length) return null;
  const out = [];
  for (let i = 0; i < lms.length; i++) {
    const p = lms[i];
    out.push({ x: p.x, y: p.y, z: p.z || 0, visibility: p.visibility });
  }
  return out;
}

const ILSA_MEDIAPIPE = "http://127.0.0.1:8765/mediapipe/";
/** Prefijo de assets: chrome-extension://… o ILSA si el XHR del sandbox falla. */
let assetBase = null;

/**
 * Nombre de archivo que pide Holistic (fuerza pose *lite*).
 * @param {string} file
 * @returns {string}
 */
function assetName(file) {
  const raw = String(file).split("/").pop();
  if (raw === "pose_landmark_full.tflite" || raw === "pose_landmark_heavy.tflite") {
    return "pose_landmark_lite.tflite";
  }
  return raw;
}

/**
 * Resuelve un asset WASM/tflite vendored.
 * @param {string} file Nombre que pide Holistic.
 * @returns {string} URL absoluta.
 */
function locateFile(file) {
  const name = assetName(file);
  if (assetBase) return assetBase + name;
  return new URL("./vendor/mediapipe/" + name, self.location.href).href;
}

/**
 * The Emscripten loader XHRs the `.data`. If the file is missing (it is in
 * .gitignore) or Chrome blocks chrome-extension:// from the sandbox, Holistic
 * never sends `ready`. Try the extension first; on failure, ILSA `/mediapipe/`.
 * @returns {Promise<void>}
 */
async function resolveAssetBase() {
  const name = "holistic_solution_packed_assets.data";
  const extBase = new URL("./vendor/mediapipe/", self.location.href).href;
  const candidates = [extBase, ILSA_MEDIAPIPE];
  let last = "";
  for (const base of candidates) {
    try {
      const res = await fetch(base + name);
      if (res.ok) {
        assetBase = base;
        return;
      }
      last = `${base + name} → HTTP ${res.status}`;
    } catch (err) {
      last = `${base + name} → ${err && err.message ? err.message : err}`;
    }
  }
  throw new Error(
    "Falta holistic_solution_packed_assets.data. En la carpeta del repo corré: py -3 packaging/fetch_extension_assets.py y recargá la extensión. " +
      last
  );
}

/**
 * Internal MediaPipe logs (I0000 / OpenGL) do not help LSA debugging.
 * @param {any[]} args
 * @returns {boolean}
 */
function isMediaPipeLog(args) {
  const s = args.map((a) => (typeof a === "string" ? a : String(a))).join(" ");
  return /^(I|W)0000\s/.test(s) || s.includes("gl_context") || s.includes("OpenGL error checking");
}

["log", "info", "debug", "warn"].forEach((method) => {
  const orig = console[method].bind(console);
  console[method] = (...args) => {
    if (isMediaPipeLog(args)) return;
    orig(...args);
  };
});

let holistic = null;
let busy = false;
let sendStarted = 0;
const input = () => document.getElementById("input");
let inputCtx = null;

/**
 * @param {object} msg
 */
function post(msg) {
  parent.postMessage(msg, "*");
}

/** @returns {CanvasRenderingContext2D|null} */
function ensureCtx() {
  const canvas = input();
  if (!canvas) return null;
  if (!inputCtx) inputCtx = canvas.getContext("2d", { alpha: false });
  return inputCtx;
}

/**
 * Instantiate Holistic (complexity 0 = lite). `onResults` is the extraction
 * of pose + hands from the `#input` canvas.
 * @returns {Promise<void>}
 */
async function init() {
  if (typeof Holistic !== "function") {
    throw new Error("No cargó holistic.js en el sandbox");
  }
  await resolveAssetBase();
  holistic = new Holistic({ locateFile });
  holistic.setOptions({
    selfieMode: false,
    modelComplexity: 0,
    smoothLandmarks: true,
    refineFaceLandmarks: false,
    enableSegmentation: false,
    enableFaceGeometry: false,
    minDetectionConfidence: 0.4,
    minTrackingConfidence: 0.4,
  });
  holistic.onResults((results) => {
    const holistic_ms = sendStarted ? Math.round(performance.now() - sendStarted) : null;
    sendStarted = 0;
    post({
      type: "landmarks",
      pose: packLandmarks(results.poseLandmarks),
      left_hand: packLandmarks(results.leftHandLandmarks),
      right_hand: packLandmarks(results.rightHandLandmarks),
      holistic_ms,
    });
  });
  if (typeof holistic.initialize === "function") {
    await holistic.initialize();
  }
  post({ type: "ready" });
}

/**
 * Draw the frame on `#input` and run `holistic.send`.
 * @param {{ buffer?: ArrayBuffer, width?: number, height?: number, bitmap?: ImageBitmap }} data
 */
async function sendFrame(data) {
  const canvas = input();
  const ctx = ensureCtx();
  if (!holistic || !canvas || !ctx) return;

  if (data.buffer && data.width && data.height) {
    const pixels = new Uint8ClampedArray(data.buffer);
    const imageData = new ImageData(pixels, data.width, data.height);
    if (canvas.width !== data.width) canvas.width = data.width;
    if (canvas.height !== data.height) canvas.height = data.height;
    inputCtx = canvas.getContext("2d", { alpha: false });
    inputCtx.putImageData(imageData, 0, 0);
  } else if (data.bitmap) {
    const img = data.bitmap;
    if (canvas.width !== img.width) canvas.width = img.width;
    if (canvas.height !== img.height) canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    if (img.close) {
      try {
        img.close();
      } catch (_) {}
    }
  } else {
    return;
  }

  sendStarted = performance.now();
  await holistic.send({ image: canvas });
}

window.addEventListener("message", async (event) => {
  if (event.source !== parent) return;
  const data = event.data;
  if (!data || !data.type) return;
  if (data.type === "frame") {
    if (!holistic || busy) {
      post({ type: "need-frame" });
      return;
    }
    busy = true;
    try {
      await sendFrame(data);
    } catch (err) {
      post({ type: "error", message: String(err && err.message ? err.message : err) });
    } finally {
      busy = false;
    }
    return;
  }
  if (data.type === "stop") {
    busy = false;
  }
});

window.addEventListener("error", (event) => {
  post({ type: "error", message: event.message || "Error en MediaPipe" });
});

window.addEventListener("unhandledrejection", (event) => {
  post({ type: "error", message: String(event.reason || "MediaPipe rechazó una promesa") });
});

init().catch((err) => {
  post({ type: "error", message: String(err && err.message ? err.message : err) });
});
