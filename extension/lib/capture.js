const FALLBACK_CFG = {
  motion_pixel_threshold: 500,
  landmark_motion_threshold: 0.008,
  static_hands_frames_to_start: 4,
  still_frames_limit: 10,
  capture_buffer_size: 60,
  missing_hands_limit: 12,
  min_capture_frames: 5,
  max_frames: 16,
  capture_mode: "auto",
  utterance_pause_sec: 4.0,
};

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

function shouldStartRecording(mode, handsPresent, isMoving, consecutiveHands, staticFrames) {
  if (!handsPresent) return false;
  if (mode === "dynamic") return isMoving;
  if (mode === "static") return consecutiveHands >= Math.min(2, staticFrames || 2);
  return true;
}

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

  function cfg() {
    return { ...FALLBACK_CFG, ...(getCfg() || {}) };
  }

  function bumpActivity(fromHands) {
    const now = Date.now() / 1000;
    lastActivity = now;
    if (fromHands && pendingGlosses) callbacks.onSigningActivity();
  }

  function flushSign() {
    const c = cfg();
    if (frames.length < c.min_capture_frames) {
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
    markPending(hasGlosses) {
      pendingGlosses = Boolean(hasGlosses);
    },
    noteInferenceActivity() {
      bumpActivity(false);
      pendingGlosses = true;
    },
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

      const start = shouldStartRecording(
        c.capture_mode,
        handsPresent,
        isMoving,
        consecutiveHands,
        c.static_hands_frames_to_start
      );
      const recording = frames.length > 0 || start;

      if (recording) {
        if (handsPresent) {
          missingHands = 0;
          missingSince = 0;
          frames.push(packFrame(results));
          if (isMoving) consecutiveStill = 0;
          else consecutiveStill += 1;
          if (frames.length >= c.capture_buffer_size) {
            flushSign();
          }
        } else if (frames.length > 0) {
          const t = Date.now() / 1000;
          if (!missingSince) missingSince = t;
          missingHands += 1;
          frames.push(frames[frames.length - 1]);
          const graceSec = Math.max(0.35, (c.missing_hands_limit || 12) / 30);
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
