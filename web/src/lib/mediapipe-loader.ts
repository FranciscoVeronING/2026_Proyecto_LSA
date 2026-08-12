/**
 * MediaPipe Holistic — singleton, todo desde CDN (misma versión en .js/.wasm/.data).
 * Evita doble init (React Strict Mode) y conflictos Emscripten Module.arguments.
 */

const HOLISTIC_VERSION = "0.5.1675471629";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${HOLISTIC_VERSION}`;

export type HolisticLandmark = { x: number; y: number; z: number };

export type HolisticResults = {
  poseLandmarks?: HolisticLandmark[];
  leftHandLandmarks?: HolisticLandmark[];
  rightHandLandmarks?: HolisticLandmark[];
};

export interface HolisticInstance {
  setOptions(options: Record<string, unknown>): void;
  onResults(callback: (results: HolisticResults) => void): void;
  initialize(): Promise<void>;
  send(input: { image: HTMLVideoElement | HTMLCanvasElement }): Promise<void>;
  close(): Promise<void>;
}

export type HolisticConstructor = new (config?: {
  locateFile?: (file: string, prefix?: string) => string;
}) => HolisticInstance;

declare global {
  interface Window {
    Holistic?: HolisticConstructor;
  }
}

let scriptPromise: Promise<void> | null = null;
let instancePromise: Promise<HolisticInstance> | null = null;
let sharedInstance: HolisticInstance | null = null;

function locateFile(file: string): string {
  return `${CDN}/${file}`;
}

function loadHolisticScript(): Promise<void> {
  if (window.Holistic) return Promise.resolve();
  if (scriptPromise) return scriptPromise;

  scriptPromise = new Promise((resolve, reject) => {
    const src = `${CDN}/holistic.js`;
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener(
        "error",
        () => reject(new Error(`No se pudo cargar ${src}`)),
        { once: true },
      );
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve();
    script.onerror = () => {
      scriptPromise = null;
      reject(new Error(`No se pudo cargar ${src}`));
    };
    document.head.appendChild(script);
  });

  return scriptPromise;
}

async function createSharedInstance(): Promise<HolisticInstance> {
  await loadHolisticScript();
  if (!window.Holistic) {
    throw new Error("MediaPipe Holistic no se registró en window");
  }

  const holistic = new window.Holistic({ locateFile });
  await holistic.initialize();
  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  sharedInstance = holistic;
  return holistic;
}

/**
 * Devuelve la instancia compartida (una sola init por sesión).
 * Actualiza onResults en cada llamada; no llamar close() al desmontar.
 */
export async function acquireHolistic(
  onResults: (results: HolisticResults) => void,
): Promise<HolisticInstance> {
  if (sharedInstance) {
    sharedInstance.onResults(onResults);
    return sharedInstance;
  }

  if (!instancePromise) {
    instancePromise = createSharedInstance().catch((err) => {
      instancePromise = null;
      sharedInstance = null;
      throw err;
    });
  }

  const holistic = await instancePromise;
  holistic.onResults(onResults);
  return holistic;
}

export function isHolisticReady(): boolean {
  return sharedInstance !== null;
}
