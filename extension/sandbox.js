/**
 * Iframe sandbox: **acá se extraen los landmarks** (MediaPipe Holistic WASM).
 *
 * El padre manda píxeles `{ type: "frame", width, height, buffer }`.
 * Holistic devuelve esqueleto; este script postea
 * `{ type: "landmarks", pose, left_hand, right_hand }` al offscreen.
 * El backend nunca ve el JPEG.
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

/**
 * Resuelve un asset WASM/tflite vendored. Fuerza el modelo *lite* de pose.
 * @param {string} file Nombre que pide Holistic.
 * @returns {string} URL absoluta chrome-extension://…/vendor/mediapipe/…
 */
function locateFile(file) {
  const raw = String(file).split("/").pop();
  const name =
    raw === "pose_landmark_full.tflite" || raw === "pose_landmark_heavy.tflite"
      ? "pose_landmark_lite.tflite"
      : raw;
  return new URL("vendor/mediapipe/" + name, location.href).href;
}

/**
 * Logs internos de MediaPipe (I0000 / OpenGL) no aportan al debug de LSA.
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
 * Instancia Holistic (complexity 0 = lite). `onResults` es la extracción
 * de pose + manos a partir del canvas `#input`.
 * @returns {Promise<void>}
 */
async function init() {
  if (typeof Holistic !== "function") {
    throw new Error("No cargó holistic.js en el sandbox");
  }
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
    post({
      type: "landmarks",
      pose: packLandmarks(results.poseLandmarks),
      left_hand: packLandmarks(results.leftHandLandmarks),
      right_hand: packLandmarks(results.rightHandLandmarks),
    });
  });
  if (typeof holistic.initialize === "function") {
    await holistic.initialize();
  }
  post({ type: "ready" });
}

/**
 * Pinta el frame en `#input` y corre `holistic.send`.
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
