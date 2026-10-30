/**
 * Temporal crop of a sign. Runs **in the extension**, with landmarks
 * Holistic already produced. The backend is not involved until `onSign` → `POST /sign`.
 *
 * Pixel motion is not used (people behind the camera in Meet). In `auto`,
 * it opens after N consecutive frames with a usable hand; it ends when the hands leave
 * (~0.4 s grace if Holistic drops a frame). It does not cut on “still hands”:
 * static signs (HOLA, numbers, etc.) still have to be recorded.
 */

/**
 * Thresholds if `/config` has not answered yet.
 * Names match `src/classifier/config.py`.
 *
 * @typedef {object} CaptureConfig
 * @property {number} landmark_motion_threshold Minimum L2 hand motion for “signing”.
 * @property {number} capture_buffer_size Max raw frames of a sign.
 * @property {number} missing_hands_limit Frames @30fps equivalent to the no-hands grace.
 * @property {number} min_capture_frames Frames with a real hand; below that it is discarded.
 * @property {number} hands_frames_to_start Consecutive frames with a hand to open (auto mode).
 * @property {number} max_frames Size TinySkeleton consumes.
 * @property {"auto"|"dynamic"|"static"} capture_mode
 * @property {number} utterance_pause_sec Silence → close utterance.
 * @property {number} static_hands_frames_to_start `static` mode only.
 */
const FALLBACK_CFG = {
  motion_pixel_threshold: 500,
  landmark_motion_threshold: 0.008,
  static_hands_frames_to_start: 2,
  hands_frames_to_start: 2,
  still_frames_limit: 14,
  capture_buffer_size: 60,
  missing_hands_limit: 12,
  min_capture_frames: 2,
  max_frames: 16,
  capture_mode: "auto",
  utterance_pause_sec: 4.0,
};

/**
 * Uniform subsample (floor indices, same criterion as an integer linspace).
 *
 * @param {object[]} frames Raw `packFrame` buffer.
 * @param {number} target Desired count (16).
 * @returns {object[]} `frames` if already short; otherwise `target` samples.
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
 * Should recording start on this frame?
 *
 * Meet uses `capture_mode: "auto"`: N consecutive frames with a usable hand are required.
 * A single Holistic ghost frame (moving sideways) does not open the sign.
 * `dynamic` does require hand L2; `static` requires N frames with hands.
 *
 * @param {string} mode `auto` | `dynamic` | `static`.
 * @param {boolean} handsPresent At least one usable hand (`anyHandPresent`).
 * @param {boolean} isMoving Hand L2 > threshold (dynamic mode only).
 * @param {number} consecutiveHands Consecutive frames with hands.
 * @param {number} startFrames Frame threshold to open (`hands_frames_to_start`).
 * @returns {boolean}
 */
function shouldStartRecording(mode, handsPresent, isMoving, consecutiveHands, startFrames) {
  if (!handsPresent) return false;
  if (mode === "dynamic") return isMoving;
  const need = Math.max(1, startFrames || 1);
  return consecutiveHands >= need;
}

/**
 * Create the capture engine. One `step` per Holistic result.
 *
 * @param {() => Partial<CaptureConfig>|null|undefined} getCfg Live config (e.g. from `/config`).
 * @param {object} callbacks
 * @param {(frames: object[]) => void} callbacks.onSign Sign ready (already subsampled).
 * @param {() => void} callbacks.onSigningActivity Hands moving with pending glosses.
 * @param {() => (void|Promise<void>)} callbacks.onUtterancePause Long pause: request Spanish.
 * @returns {{
 *   markPending: (hasGlosses: boolean) => void,
 *   noteInferenceActivity: () => void,
 *   forgetPrevHand: () => void,
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
  let pauseStartLogged = false;
  let handsSince = 0;
  let stillSince = 0;
  let captureStartedAt = 0;
  let nHandTicks = 0;
  let signEndedAtMs = 0;

  /** @returns {CaptureConfig} */
  function cfg() {
    return { ...FALLBACK_CFG, ...(getCfg() || {}) };
  }

  /**
   * Mark activity for the utterance clock.
   * @param {boolean} fromHands If it comes from real motion, notify the backend (`/activity`).
   */
  function bumpActivity(fromHands) {
    const now = Date.now() / 1000;
    lastActivity = now;
    pauseStartLogged = false;
    if (fromHands && pendingGlosses) callbacks.onSigningActivity();
  }

  /**
   * Close the current sign: discard if there are few real hands; otherwise subsample.
   * @returns {void}
   */
  function flushSign() {
    const c = cfg();
    const nGood = nHandTicks;
    if (nGood < Math.max(1, c.min_capture_frames || 2)) {
      frames = [];
      consecutiveStill = 0;
      missingHands = 0;
      missingSince = 0;
      stillSince = 0;
      captureStartedAt = 0;
      nHandTicks = 0;
      return;
    }
    const nRaw = frames.length;
    const payload = uniformSampleFrames(frames.slice(), c.max_frames || 16);
    const capture_ms = captureStartedAt ? Date.now() - captureStartedAt : null;
    const time_to_record_ms =
      captureStartedAt && handsSince ? Math.max(0, captureStartedAt - handsSince) : null;
    frames = [];
    consecutiveStill = 0;
    missingHands = 0;
    missingSince = 0;
    stillSince = 0;
    captureStartedAt = 0;
    nHandTicks = 0;
    callbacks.onSign(payload, {
      capture_ms,
      n_raw: nRaw,
      n_sent: payload.length,
      time_to_record_ms,
    });
  }

  return {
    /**
     * @param {boolean} hasGlosses There are glosses in the backend buffer.
     */
    markPending(hasGlosses) {
      pendingGlosses = Boolean(hasGlosses);
    },
    /** After an accepted `/sign`: the utterance is still alive. */
    noteInferenceActivity() {
      bumpActivity(false);
      pendingGlosses = true;
    },
    /**
     * Drop the last motion vector (e.g. handedness flipped). Do not clear the
     * utterance pause clock: that would skip `/utterance/end` and the caption.
     */
    forgetPrevHand() {
      prevHand = null;
    },
    /**
     * One Holistic tick.
     *
     * @param {object} args
     * @param {object} args.results `poseLandmarks` / `leftHandLandmarks` / `rightHandLandmarks`.
     * @param {boolean} args.leftHanded Mirror the vector if the user is left-handed.
     * @returns {{ recording: boolean, bufferLen: number, still: number, handsPresent: boolean, moving: boolean }}
     */
    step({ results, video, motionCtx, leftHanded }) {
      const c = cfg();
      const handsPresent = anyHandPresent(results);
      const vector = handsPresent ? extractNormalizedVector(results, leftHanded) : null;
      const landmarkMotion = vector ? handMotion(vector, prevHand) : 0;
      if (vector) prevHand = vector;
      else prevHand = null;

      if (handsPresent) {
        consecutiveHands += 1;
        if (!handsSince) handsSince = Date.now();
      } else {
        consecutiveHands = 0;
        handsSince = 0;
        if (pendingGlosses && lastActivity > 0 && !pauseStartLogged) {
          pauseStartLogged = true;
          signEndedAtMs = Date.now();
          console.log(
            "[metrics] Tiempo de Pausa (Heurística de Cierre): t_pausa_inicio — NOT model time"
          );
        }
      }

      const isMoving = landmarkMotion > c.landmark_motion_threshold;

      const startNeed = c.hands_frames_to_start || c.static_hands_frames_to_start || 2;
      const start = shouldStartRecording(
        c.capture_mode,
        handsPresent,
        isMoving,
        consecutiveHands,
        startNeed
      );
      const recording = frames.length > 0 || start;
      if (start && frames.length === 0) captureStartedAt = Date.now();

      if (recording) {
        if (handsPresent) {
          missingHands = 0;
          missingSince = 0;
          frames.push(packFrame(results));
          nHandTicks += 1;
          const cap = c.capture_buffer_size || 60;
          if (frames.length > cap) frames.shift();
          bumpActivity(isMoving);
        } else if (frames.length > 0) {
          const t = Date.now() / 1000;
          if (!missingSince) missingSince = t;
          missingHands += 1;
          frames.push(frames[frames.length - 1]);
          const graceSec = Math.max(0.2, (c.missing_hands_limit || 6) / 30);
          if (t - missingSince >= graceSec) flushSign();
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
        const pause_ms = Math.round((now - lastActivity) * 1000);
        console.log(
          "[metrics] Tiempo de Pausa (Heurística de Cierre): t_pausa_fin pause_ms=" +
            pause_ms +
            " — NOT model time"
        );
        Promise.resolve(
          callbacks.onUtterancePause({
            pause_ms,
            sign_end_ms: signEndedAtMs || Math.round(lastActivity * 1000),
          })
        )
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
    /** Clear buffer and clocks (Meet stop / new session). */
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
      pauseStartLogged = false;
      handsSince = 0;
      stillSince = 0;
      captureStartedAt = 0;
      nHandTicks = 0;
      signEndedAtMs = 0;
    },
  };
}
