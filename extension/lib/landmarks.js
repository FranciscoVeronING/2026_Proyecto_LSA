const POSE_DIM = 33 * 3;
const HAND_DIM = 21 * 3;

function lmList(landmarks) {
  if (!landmarks || !landmarks.length) return null;
  return landmarks.map((p) => [p.x, p.y, p.z || 0]);
}

function packFrame(results) {
  return {
    pose: lmList(results.poseLandmarks),
    left_hand: lmList(results.leftHandLandmarks),
    right_hand: lmList(results.rightHandLandmarks),
  };
}

function handIsPresent(lms) {
  if (!lms || lms.length < 15) return false;
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  let usable = 0;
  for (let i = 0; i < lms.length; i++) {
    const p = lms[i];
    if (!p) continue;
    const x = p.x;
    const y = p.y;
    if (x < -0.2 || x > 1.2 || y < -0.2 || y > 1.2) continue;
    if (Math.abs(x) < 1e-5 && Math.abs(y) < 1e-5) continue;
    usable += 1;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (usable < 12) return false;
  return maxX - minX > 0.04 || maxY - minY > 0.04;
}

function anyHandPresent(results) {
  return handIsPresent(results.leftHandLandmarks) || handIsPresent(results.rightHandLandmarks);
}

function createLandmarkSmoother(alpha) {
  const a = alpha == null ? 0.6 : alpha;
  const prev = { pose: null, left: null, right: null };

  function blend(key, lms) {
    if (!lms || !lms.length) {
      prev[key] = null;
      return lms;
    }
    const last = prev[key];
    const out = [];
    const stored = new Float32Array(lms.length * 3);
    for (let i = 0; i < lms.length; i++) {
      const p = lms[i];
      const x = p.x;
      const y = p.y;
      const z = p.z || 0;
      let sx = x;
      let sy = y;
      let sz = z;
      if (last && last.length === stored.length) {
        sx = a * x + (1 - a) * last[i * 3];
        sy = a * y + (1 - a) * last[i * 3 + 1];
        sz = a * z + (1 - a) * last[i * 3 + 2];
      }
      stored[i * 3] = sx;
      stored[i * 3 + 1] = sy;
      stored[i * 3 + 2] = sz;
      out.push({ x: sx, y: sy, z: sz, visibility: p.visibility });
    }
    prev[key] = stored;
    return out;
  }

  return {
    reset() {
      prev.pose = null;
      prev.left = null;
      prev.right = null;
    },
    apply(results) {
      return {
        poseLandmarks: blend("pose", results.poseLandmarks),
        leftHandLandmarks: blend("left", results.leftHandLandmarks),
        rightHandLandmarks: blend("right", results.rightHandLandmarks),
      };
    },
  };
}

function zeros(n) {
  return new Float32Array(n);
}

function flattenLandmarks(landmarks, expected) {
  const out = zeros(expected * 3);
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

function countPixelMotion(prevGray, gray, threshold) {
  if (!prevGray || prevGray.length !== gray.length) return false;
  let changed = 0;
  for (let i = 0; i < gray.length; i++) {
    if (Math.abs(gray[i] - prevGray[i]) > 25) changed += 1;
  }
  return changed > threshold;
}

function videoToGray(ctx, video, w, h) {
  ctx.drawImage(video, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  const gray = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (img.data[p] + img.data[p + 1] + img.data[p + 2]) / 3;
  }
  return gray;
}

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28], [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22],
];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function _pt(lm, i, w, h) {
  const p = lm && lm[i];
  if (!p) return null;
  return [p.x * w, p.y * h];
}

function _strokeGroup(ctx, lm, connections, color, w, h) {
  if (!lm || !lm.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of connections) {
    const pa = _pt(lm, a, w, h);
    const pb = _pt(lm, b, w, h);
    if (!pa || !pb) continue;
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
  }
  ctx.stroke();
  ctx.fillStyle = color;
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawCameraAndLandmarks(ctx, source, results) {
  const w = source.videoWidth || source.width;
  const h = source.videoHeight || source.height;
  if (!w || !h) return;
  const canvas = ctx.canvas;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(source, 0, 0, w, h);
  if (!results) return;
  _strokeGroup(ctx, results.poseLandmarks, POSE_CONNECTIONS, "rgba(34,211,238,0.9)", w, h);
  _strokeGroup(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, "rgba(74,222,128,0.95)", w, h);
  _strokeGroup(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, "rgba(250,204,21,0.95)", w, h);
}
