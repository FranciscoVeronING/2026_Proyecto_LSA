/**
 * Packing and vectorization of landmarks **already extracted** by Holistic.
 *
 * This file does not look at pixels. The skeleton comes from `sandbox.js` (MediaPipe).
 * Here we decide if a hand is usable, build the `POST /sign` JSON, and the
 * 225-d vector used to measure L2 motion in `capture.js`.
 *
 * One frame for the classifier:
 *   pose 33×3  +  left hand 21×3  +  right hand 21×3  =  225 floats.
 */

/** Dimension of the pose block in the flat vector. */
const POSE_DIM = 33 * 3;
/** Dimension of one hand. */
const HAND_DIM = 21 * 3;

/**
 * Serialize a list of MediaPipe points to `[x,y,z]` arrays.
 * @param {Array<{x:number,y:number,z?:number}>|null|undefined} landmarks
 * @returns {number[][]|null} `null` if there are no points (the backend treats zeros).
 */
function lmList(landmarks) {
  if (!landmarks || !landmarks.length) return null;
  return landmarks.map((p) => [p.x, p.y, p.z || 0]);
}

/**
 * Pack a Holistic result for `POST /sign`.
 * The backend receives these points, not the video.
 * @param {{ poseLandmarks?: object[], leftHandLandmarks?: object[], rightHandLandmarks?: object[] }} results
 * @returns {{ pose: number[][]|null, left_hand: number[][]|null, right_hand: number[][]|null }}
 */
function packFrame(results) {
  return {
    pose: lmList(results.poseLandmarks),
    left_hand: handIsPresent(results.leftHandLandmarks, poseWrist(results.poseLandmarks, "left"))
      ? lmList(results.leftHandLandmarks)
      : null,
    right_hand: handIsPresent(results.rightHandLandmarks, poseWrist(results.poseLandmarks, "right"))
      ? lmList(results.rightHandLandmarks)
      : null,
  };
}

/**
 * Is this hand usable? Holistic invents hands when the torso moves:
 * points at (0,0), a tiny bbox, or far from the pose wrist.
 * Holistic hands have no `visibility` (only x, y, z).
 *
 * @param {Array<{x:number,y:number}>|number[][]|null|undefined} lms
 * @param {{x:number,y:number,visibility?:number}|number[]|null|undefined} [wrist] Pose wrist.
 * @returns {boolean}
 */
function xyOf(p) {
  if (!p) return null;
  if (Array.isArray(p)) return { x: +p[0], y: +p[1], visibility: p[3] };
  return { x: p.x, y: p.y, visibility: p.visibility };
}

function handIsPresent(lms, wrist) {
  if (!lms || lms.length < 18) return false;
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  let usable = 0;
  for (let i = 0; i < lms.length; i++) {
    const p = xyOf(lms[i]);
    if (!p) continue;
    const x = p.x;
    const y = p.y;
    if (x < -0.15 || x > 1.15 || y < -0.15 || y > 1.15) continue;
    if (Math.abs(x) < 1e-5 && Math.abs(y) < 1e-5) continue;
    usable += 1;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (usable < 14) return false;
  const bw = maxX - minX;
  const bh = maxY - minY;
  if (bw < 0.035 && bh < 0.035) return false;
  if (bw > 0.85 || bh > 0.85) return false;
  if (wrist) {
    const w = xyOf(wrist);
    const hw = xyOf(lms[0]);
    if (w && hw) {
      const poseWristOk = typeof w.visibility !== "number" || w.visibility >= 0.2;
      const dist = Math.hypot(hw.x - w.x, hw.y - w.y);
      if (poseWristOk && dist > 0.42) return false;
    }
  }
  return true;
}

function poseWrist(pose, side) {
  if (!pose || pose.length < 17) return null;
  return pose[side === "left" ? 15 : 16] || null;
}

/**
 * @param {{ poseLandmarks?: object[], leftHandLandmarks?: object[], rightHandLandmarks?: object[] }} results
 * @returns {boolean}
 */
function anyHandPresent(results) {
  const pose = results.poseLandmarks;
  const leftOk = handIsPresent(results.leftHandLandmarks, poseWrist(pose, "left"));
  const rightOk = handIsPresent(results.rightHandLandmarks, poseWrist(pose, "right"));
  return leftOk || rightOk;
}

/**
 * List of points → flat vector `expected * 3`.
 * `new Float32Array(n)` is already zeros; no extra helper is needed.
 *
 * @param {Array<{x:number,y:number,z?:number}>|null|undefined} landmarks
 * @param {number} expected 33 (pose) or 21 (hand).
 * @returns {Float32Array}
 */
function flattenLandmarks(landmarks, expected) {
  const out = new Float32Array(expected * 3);
  if (!landmarks || !landmarks.length) return out;
  for (let i = 0; i < expected; i++) {
    const p = landmarks[i];
    if (!p) continue;
    out[i * 3] = p.x;
    out[i * 3 + 1] = p.y;
    out[i * 3 + 2] = p.z || 0;
  }
  return out;
}

/**
 * Position/size invariance: (point - anchor) / shoulder distance.
 * @param {Float32Array} flat
 * @param {number[]} anchor `[x,y,z]` midpoint of the shoulders.
 * @param {number} scale Inter-shoulder distance.
 * @returns {Float32Array}
 */
function normalizeSpatial(flat, anchor, scale) {
  const out = new Float32Array(flat.length);
  let allZero = true;
  for (let i = 0; i < flat.length; i++) {
    if (flat[i] !== 0) allZero = false;
  }
  if (allZero) return flat;
  for (let i = 0; i < flat.length; i += 3) {
    out[i] = (flat[i] - anchor[0]) / scale;
    out[i + 1] = (flat[i + 1] - anchor[1]) / scale;
    out[i + 2] = (flat[i + 2] - anchor[2]) / scale;
  }
  return out;
}

/**
 * One Holistic frame → 225-d vector aligned with the classifier.
 * If `leftHanded`, mirror X and swap hand blocks (the model was trained right-handed).
 *
 * @param {object} results Holistic landmarks.
 * @param {boolean} leftHanded
 * @returns {Float32Array} Length `POSE_DIM + 2 * HAND_DIM`.
 */
function extractNormalizedVector(results, leftHanded) {
  const pose = results.poseLandmarks;
  let anchor = [0, 0, 0];
  let scale = 1;
  if (pose && pose[11] && pose[12]) {
    const ls = pose[11];
    const rs = pose[12];
    anchor = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2, ((ls.z || 0) + (rs.z || 0)) / 2];
    scale = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    if (scale < 1e-5) scale = 1;
  }
  const poseV = normalizeSpatial(flattenLandmarks(pose, 33), anchor, scale);
  const lh = normalizeSpatial(flattenLandmarks(results.leftHandLandmarks, 21), anchor, scale);
  const rh = normalizeSpatial(flattenLandmarks(results.rightHandLandmarks, 21), anchor, scale);
  const vector = new Float32Array(POSE_DIM + HAND_DIM * 2);
  vector.set(poseV, 0);
  vector.set(lh, POSE_DIM);
  vector.set(rh, POSE_DIM + HAND_DIM);
  if (leftHanded) {
    for (let i = 0; i < vector.length; i += 3) vector[i] *= -1;
    const left = vector.slice(POSE_DIM, POSE_DIM + HAND_DIM);
    const right = vector.slice(POSE_DIM + HAND_DIM, POSE_DIM + HAND_DIM * 2);
    vector.set(right, POSE_DIM);
    vector.set(left, POSE_DIM + HAND_DIM);
  }
  return vector;
}

/**
 * Euclidean distance (L2 norm) between the hands of two frames:
 * √(Σ (current − previous)²) over both hands' coordinates.
 *
 * Does not look at camera pixels. In `auto` mode (Meet) this does **not** open the sign:
 * start is “there is a usable hand” (`shouldStartRecording`). This value
 * feeds `/activity` and `dynamic` mode.
 *
 * @param {Float32Array} current
 * @param {Float32Array|null} previous
 * @returns {number} 0 if there is no previous or a vector is all zeros.
 */
function handMotion(current, previous) {
  if (!previous) return 0;
  let sum = 0;
  let currZero = true;
  let prevZero = true;
  for (let i = POSE_DIM; i < current.length; i++) {
    const d = current[i] - previous[i];
    sum += d * d;
    if (current[i] !== 0) currZero = false;
    if (previous[i] !== 0) prevZero = false;
  }
  if (currZero || prevZero) return 0;
  return Math.sqrt(sum);
}
