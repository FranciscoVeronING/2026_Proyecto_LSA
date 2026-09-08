/**
 * Empaquetado y vectorización de landmarks **ya extraídos** por Holistic.
 *
 * Este archivo no mira píxeles. El esqueleto sale de `sandbox.js` (MediaPipe).
 * Acá se decide si una mano es usable, se arma el JSON de `POST /sign` y el
 * vector 225 para medir movimiento L2 en `capture.js`.
 *
 * Un frame para el clasificador:
 *   pose 33×3  +  mano izq. 21×3  +  mano der. 21×3  =  225 floats.
 */

/** Dimensión del bloque de pose en el vector plano. */
const POSE_DIM = 33 * 3;
/** Dimensión de una mano. */
const HAND_DIM = 21 * 3;

/**
 * Serializa una lista de puntos MediaPipe a arrays `[x,y,z]`.
 * @param {Array<{x:number,y:number,z?:number}>|null|undefined} landmarks
 * @returns {number[][]|null} `null` si no hay puntos (el backend interpreta ceros).
 */
function lmList(landmarks) {
  if (!landmarks || !landmarks.length) return null;
  return landmarks.map((p) => [p.x, p.y, p.z || 0]);
}

/**
 * Empaqueta un resultado Holistic para `POST /sign`.
 * El backend recibe estos puntos, no el video.
 * @param {{ poseLandmarks?: object[], leftHandLandmarks?: object[], rightHandLandmarks?: object[] }} results
 * @returns {{ pose: number[][]|null, left_hand: number[][]|null, right_hand: number[][]|null }}
 */
function packFrame(results) {
  return {
    pose: lmList(results.poseLandmarks),
    left_hand: lmList(results.leftHandLandmarks),
    right_hand: lmList(results.rightHandLandmarks),
  };
}

/**
 * ¿Esta mano es usable? No alcanza con “el array no está vacío”: Holistic a
 * veces devuelve puntos en (0,0) o fuera de cuadro.
 *
 * @param {Array<{x:number,y:number}>|null|undefined} lms 21 puntos o menos.
 * @returns {boolean} true si hay ≥12 puntos razonables y un bbox mínimo.
 */
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

/**
 * @param {{ leftHandLandmarks?: object[], rightHandLandmarks?: object[] }} results
 * @returns {boolean}
 */
function anyHandPresent(results) {
  return handIsPresent(results.leftHandLandmarks) || handIsPresent(results.rightHandLandmarks);
}

/**
 * Lista de puntos → vector plano `expected * 3`.
 * `new Float32Array(n)` ya nace lleno de ceros; no hace falta un helper extra.
 *
 * @param {Array<{x:number,y:number,z?:number}>|null|undefined} landmarks
 * @param {number} expected 33 (pose) o 21 (mano).
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
 * Invarianza a posición/tamaño: (punto - ancla) / distancia entre hombros.
 * @param {Float32Array} flat
 * @param {number[]} anchor `[x,y,z]` punto medio de hombros.
 * @param {number} scale Distancia inter-hombros.
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
 * Un frame Holistic → vector 225 alineado con el clasificador.
 * Si `leftHanded`, espeja X y cruza bloques de manos (el modelo se entrenó diestro).
 *
 * @param {object} results Landmarks Holistic.
 * @param {boolean} leftHanded
 * @returns {Float32Array} Longitud `POSE_DIM + 2 * HAND_DIM`.
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
 * Distancia euclídea (norma L2) entre las manos de dos frames:
 * √(Σ (actual − anterior)²) sobre las coordenadas de ambas manos.
 *
 * No mira píxeles de la cámara. En modo `auto` (Meet) **no** abre la seña:
 * el arranque es “hay mano usable” (`shouldStartRecording`). Este valor
 * alimenta `/activity` y el modo `dynamic`.
 *
 * @param {Float32Array} current
 * @param {Float32Array|null} previous
 * @returns {number} 0 si falta previo o algún vector es todo ceros.
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
