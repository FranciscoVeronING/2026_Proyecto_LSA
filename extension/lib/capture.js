/**
 * Recorte temporal de una seña. Corre **en la extensión**, con landmarks
 * que ya trajo Holistic. El backend no participa hasta `onSign` → `POST /sign`.
 *
 * No se usa movimiento de píxeles (gente detrás de cámara en Meet). En `auto`,
 * empieza tras varios frames seguidos con una mano usable (no un fantasma de
 * Holistic al mover el torso); termina si las manos quedan quietas ~28 frames,
 * a los 60 frames, o ~0,4 s sin manos.
 */

/**
 * Umbrales por si `/config` todavía no respondió.
 * Los nombres coinciden con `src/classifier/config.py`.
 *
 * @typedef {object} CaptureConfig
 * @property {number} landmark_motion_threshold Movimiento minimo L2 de manos para “está señando”.
 * @property {number} capture_buffer_size Máximo de frames crudos de una seña.
 * @property {number} missing_hands_limit Frames @30fps equivalentes a la gracia sin manos.
 * @property {number} min_capture_frames Frames con mano real; por debajo se descarta.
 * @property {number} hands_frames_to_start Seguidos con mano para abrir (modo auto).
 * @property {number} max_frames Tamaño que consume TinySkeleton.
 * @property {"auto"|"dynamic"|"static"} capture_mode
 * @property {number} utterance_pause_sec Silencio → cerrar enunciado.
 * @property {number} static_hands_frames_to_start Solo modo `static`.
 */
const FALLBACK_CFG = {
  motion_pixel_threshold: 500,
  landmark_motion_threshold: 0.008,
  static_hands_frames_to_start: 4,
  hands_frames_to_start: 6,
  still_frames_limit: 28,
  capture_buffer_size: 60,
  missing_hands_limit: 12,
  min_capture_frames: 8,
  max_frames: 16,
  capture_mode: "auto",
  utterance_pause_sec: 4.0,
};

/**
 * Submuestreo uniforme (índices `floor`, mismo criterio que un linspace entero).
 *
 * @param {object[]} frames Buffer crudo de `packFrame`.
 * @param {number} target Cantidad deseada (16).
 * @returns {object[]} `frames` si ya es corto; si no, `target` muestras.
 */
function uniformSampleFrames(frames, target) {
  const n = frames.length;
  if (n === 0 || n <= target) return frames;
  if (target <= 1) return [frames[0]];
  const out = [];
  for (let i = 0; i < target; i++) {
    const idx = Math.floor((i * (n - 1)) / (target - 1));
    out.push(frames[idx]);
  }
  return out;
}

/**
 * ¿Hay que empezar a grabar este frame?
 *
 * Meet usa `capture_mode: "auto"`: hace falta N frames seguidos con mano usable.
 * Un solo frame fantasma de Holistic (moverse de lado) no abre la seña.
 * `dynamic` sí exige L2 de manos; `static` exige N frames con manos.
 *
 * @param {string} mode `auto` | `dynamic` | `static`.
 * @param {boolean} handsPresent Al menos una mano usable (`anyHandPresent`).
 * @param {boolean} isMoving L2 de manos > umbral (solo modo dynamic).
 * @param {number} consecutiveHands Frames seguidos con manos.
 * @param {number} startFrames Umbral de frames para abrir (`hands_frames_to_start`).
 * @returns {boolean}
 */
function shouldStartRecording(mode, handsPresent, isMoving, consecutiveHands, startFrames) {
  if (!handsPresent) return false;
  const need = Math.max(3, startFrames || 6);
  if (mode === "dynamic") return isMoving && consecutiveHands >= need;
  if (mode === "static") return consecutiveHands >= Math.max(2, startFrames || 4);
  return consecutiveHands >= need;
}

/**
 * Crea el motor de captura. Un `step` por cada resultado de Holistic.
 *
 * @param {() => Partial<CaptureConfig>|null|undefined} getCfg Config viva (p. ej. la de `/config`).
 * @param {object} callbacks
 * @param {(frames: object[]) => void} callbacks.onSign Seña lista (ya submuestreada).
 * @param {() => void} callbacks.onSigningActivity Manos en movimiento con glosas pendientes.
 * @param {() => (void|Promise<void>)} callbacks.onUtterancePause Pausa larga: pedir español.
 * @returns {{
 *   markPending: (hasGlosses: boolean) => void,
 *   noteInferenceActivity: () => void,
 *   step: (args: { results: object, video?: HTMLVideoElement, motionCtx?: CanvasRenderingContext2D, leftHanded: boolean }) => object,
 *   reset: () => void
 * }}
 */
function createCaptureEngine(getCfg, callbacks) {
  let frames = [];
  let prevHand = null;
  let consecutiveStill = 0;
  let consecutiveHands = 0;
  let missingHands = 0;
  let missingSince = 0;
  let lastActivity = 0;
  let pendingGlosses = false;
  let closing = false;

  /** @returns {CaptureConfig} */
  function cfg() {
    return { ...FALLBACK_CFG, ...(getCfg() || {}) };
  }

  /**
   * Marca actividad para el reloj de enunciado.
   * @param {boolean} fromHands Si viene de movimiento real, avisa al backend (`/activity`).
   */
  function bumpActivity(fromHands) {
    const now = Date.now() / 1000;
    lastActivity = now;
    if (fromHands && pendingGlosses) callbacks.onSigningActivity();
  }

  function countHandFrames(list) {
    let n = 0;
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if ((f.left_hand && f.left_hand.length) || (f.right_hand && f.right_hand.length)) n += 1;
    }
    return n;
  }

  /**
   * Cierra la seña actual: descarta si hay pocas manos reales; si no, submuestrea.
   * @returns {void}
   */
  function flushSign() {
    const c = cfg();
    const nGood = countHandFrames(frames);
    if (nGood < c.min_capture_frames) {
      frames = [];
      consecutiveStill = 0;
      missingHands = 0;
      missingSince = 0;
      return;
    }
    const payload = uniformSampleFrames(frames.slice(), c.max_frames || 16);
    frames = [];
    consecutiveStill = 0;
    missingHands = 0;
    missingSince = 0;
    callbacks.onSign(payload);
  }

  return {
    /**
     * @param {boolean} hasGlosses Hay glosas en el buffer del backend.
     */
    markPending(hasGlosses) {
      pendingGlosses = Boolean(hasGlosses);
    },
    /** Tras un `/sign` aceptado: el enunciado sigue vivo. */
    noteInferenceActivity() {
      bumpActivity(false);
      pendingGlosses = true;
    },
    /**
     * Un tick de Holistic.
     *
     * @param {object} args
     * @param {object} args.results `poseLandmarks` / `leftHandLandmarks` / `rightHandLandmarks`.
     * @param {boolean} args.leftHanded Espeja el vector si el usuario es zurdo.
     * @returns {{ recording: boolean, bufferLen: number, still: number, handsPresent: boolean, moving: boolean }}
     */
    step({ results, video, motionCtx, leftHanded }) {
      const c = cfg();
      const handsPresent = anyHandPresent(results);
      const vector = handsPresent ? extractNormalizedVector(results, leftHanded) : null;
      const landmarkMotion = vector ? handMotion(vector, prevHand) : 0;
      if (vector) prevHand = vector;
      else prevHand = null;

      if (handsPresent) consecutiveHands += 1;
      else consecutiveHands = 0;

      const isMoving = landmarkMotion > c.landmark_motion_threshold;
      if (handsPresent && isMoving) {
        bumpActivity(true);
      }

      const startNeed = c.hands_frames_to_start || c.static_hands_frames_to_start || 6;
      const start = shouldStartRecording(
        c.capture_mode,
        handsPresent,
        isMoving,
        consecutiveHands,
        startNeed
      );
      const recording = frames.length > 0 || start;

      if (recording) {
        if (handsPresent) {
          missingHands = 0;
          missingSince = 0;
          frames.push(packFrame(results));
          if (isMoving) consecutiveStill = 0;
          else consecutiveStill += 1;
          const stillLimit = c.still_frames_limit || 28;
          if (frames.length >= c.capture_buffer_size || consecutiveStill >= stillLimit) {
            flushSign();
          }
        } else if (frames.length > 0) {
          if (countHandFrames(frames) < c.min_capture_frames) {
            frames = [];
            consecutiveStill = 0;
            missingHands = 0;
            missingSince = 0;
          } else {
            const t = Date.now() / 1000;
            if (!missingSince) missingSince = t;
            missingHands += 1;
            frames.push(frames[frames.length - 1]);
            const graceSec = Math.max(0.35, (c.missing_hands_limit || 12) / 30);
            if (t - missingSince >= graceSec) flushSign();
          }
        }
      }

      const now = Date.now() / 1000;
      if (
        pendingGlosses &&
        !closing &&
        lastActivity > 0 &&
        now - lastActivity >= c.utterance_pause_sec
      ) {
        closing = true;
        Promise.resolve(callbacks.onUtterancePause())
          .catch(() => {})
          .finally(() => {
            closing = false;
            pendingGlosses = false;
            lastActivity = 0;
          });
      }

      return {
        recording: frames.length > 0,
        bufferLen: frames.length,
        still: consecutiveStill,
        handsPresent,
        moving: isMoving,
      };
    },
    /** Limpia buffer y relojes (stop de Meet / nueva sesión). */
    reset() {
      frames = [];
      prevHand = null;
      consecutiveStill = 0;
      consecutiveHands = 0;
      missingHands = 0;
      missingSince = 0;
      lastActivity = 0;
      pendingGlosses = false;
      closing = false;
    },
  };
}
