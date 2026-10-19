/**
 * Esqueleto de PoseVisualizer en el canvas de Meet (mundo MAIN).
 * Cada dedo tiene su color. `inject-gum.js` espeja el panel al dibujar.
 */
(function () {
  if (window.LsaAvatar) return;

  const POSE_EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7],
  ];
  // Cuerpo y manos de los primeros cuadros de HOLA.pose, en anchos de hombro,
  // con el centro de los hombros en el origen. La mano sigue el antebrazo
  // (muñeca recta): casi horizontal, un poco hacia abajo.
  const IDLE_P = [0.5,-0.019,-0.5,0.019,0.752,0.764,-0.785,0.822,0.265,0.918,-0.289,0.904,0.315,1.296,-0.316,1.293];
  const IDLE_L = [0.25,0.917,0.12,0.88,0.007,0.883,-0.069,0.899,-0.133,0.921,0.009,0.939,-0.089,0.975,-0.163,1.007,-0.223,1.03,0.025,0.978,-0.082,1.018,-0.161,1.048,-0.235,1.07,0.047,1.013,-0.06,1.056,-0.135,1.086,-0.205,1.108,0.07,1.044,-0.011,1.083,-0.074,1.105,-0.131,1.123];
  const IDLE_R = [-0.299,0.888,-0.201,0.827,-0.093,0.801,-0.001,0.788,0.067,0.763,-0.083,0.813,0.048,0.829,0.118,0.846,0.169,0.86,-0.094,0.873,0.045,0.915,0.116,0.943,0.165,0.964,-0.111,0.933,0.01,0.981,0.076,1.008,0.121,1.025,-0.133,0.987,-0.041,1.024,0.007,1.043,0.045,1.052];
  const BODY = "#d5dde6";
  const SIGN_FACE = "rgba(214, 218, 224, 0.92)";
  // Cada dedo un color, igual en las dos manos, para leer la forma.
  const FINGER = ["#f4f4f4", "#ff5c5c", "#ff9a32", "#ffe14a", "#3ddc84", "#4aa3ff"];
  const INTRO = 6 / 30;
  const FINGER_EDGES = [
    [[5, 9], [9, 13], [13, 17]],
    [[0, 1], [1, 2], [2, 3], [3, 4]],
    [[0, 5], [5, 6], [6, 7], [7, 8]],
    [[0, 9], [9, 10], [10, 11], [11, 12]],
    [[0, 13], [13, 14], [14, 15], [15, 16]],
    [[0, 17], [17, 18], [18, 19], [19, 20]],
  ];
  const FINGER_POINTS = [
    [0],
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ];

  function smooth(t) {
    const x = Math.max(0, Math.min(1, t));
    return x * x * (3 - 2 * x);
  }

  function placeIdle(rel, cx, cy, shoulder) {
    const out = new Array(rel.length);
    for (let i = 0; i < rel.length; i += 2) {
      out[i] = cx + rel[i] * shoulder;
      out[i + 1] = cy + rel[i + 1] * shoulder;
    }
    return out;
  }

  /**
   * Espera tomada de HOLA.pose. Izquierda anatómica a la derecha de la imagen.
   * @returns {{p:number[], l:number[], r:number[]}}
   */
  function idlePose(spec) {
    const w = (spec && spec.w) || 720;
    const h = (spec && spec.h) || 780;
    const shoulder = (spec && spec.shoulder) || 250;
    const shoulderY = (spec && spec.shoulder_y) || 0.55;
    const cx = w / 2;
    const cy = h * shoulderY;
    const apart = shoulder * 0.12;
    const p = placeIdle(IDLE_P, cx, cy, shoulder);
    const l = placeIdle(IDLE_L, cx, cy, shoulder);
    const r = placeIdle(IDLE_R, cx, cy, shoulder);
    p[8] += apart;
    p[10] -= apart;
    for (let i = 0; i < l.length; i += 2) l[i] += apart;
    for (let i = 0; i < r.length; i += 2) r[i] -= apart;
    return { p, l, r };
  }

  function lerpList(a, b, t) {
    const n = Math.max((a && a.length) || 0, (b && b.length) || 0);
    const out = new Array(n);
    for (let i = 0; i < n; i += 2) {
      const ax = a ? a[i] : -1;
      const ay = a ? a[i + 1] : -1;
      const bx = b ? b[i] : -1;
      const by = b ? b[i + 1] : -1;
      const aok = ax >= 0 && ay >= 0;
      const bok = bx >= 0 && by >= 0;
      if (aok && bok) {
        out[i] = ax + (bx - ax) * t;
        out[i + 1] = ay + (by - ay) * t;
      } else if (bok && t >= 0.5) {
        out[i] = bx;
        out[i + 1] = by;
      } else if (aok) {
        out[i] = ax;
        out[i + 1] = ay;
      } else {
        out[i] = -1;
        out[i + 1] = -1;
      }
    }
    return out;
  }

  function start(clip) {
    const frames = (clip && clip.frames) || [];
    return {
      frames,
      fps: Number(clip.fps) > 0 ? Number(clip.fps) : 30,
      w: clip.w || 720,
      h: clip.h || 780,
      shoulder: clip.shoulder || 250,
      shoulder_y: clip.shoulder_y || 0.55,
      faceEdges: clip.face_edges || [],
      t0: performance.now(),
    };
  }

  function durationMs(play) {
    if (!play || !play.frames || !play.frames.length) return 0;
    const span = INTRO + play.frames.length / play.fps + INTRO;
    return span * 1000;
  }

  function sample(play, now) {
    const spec = play || { w: 720, h: 780, shoulder: 250, shoulder_y: 0.55 };
    const idle = idlePose(spec);
    const rest = (alpha) => ({
      pose: idle,
      face: null,
      faceAlpha: 0,
      edges: null,
      restAlpha: alpha,
      spec,
    });
    if (!play || !play.frames || !play.frames.length) return rest(1);
    const t = (now - play.t0) / 1000;
    const playSpan = play.frames.length / play.fps;
    const end = INTRO + playSpan + INTRO;
    if (t >= end) return Object.assign(rest(1), { done: true });
    const first = play.frames[0];
    const last = play.frames[play.frames.length - 1];
    if (t < INTRO) {
      const u = smooth(t / INTRO);
      return {
        pose: {
          p: lerpList(idle.p, first.p, u),
          l: lerpList(idle.l, first.l, u),
          r: lerpList(idle.r, first.r, u),
        },
        face: first.f,
        faceAlpha: u,
        edges: play.faceEdges,
        restAlpha: 1 - u,
        spec,
      };
    }
    if (t < INTRO + playSpan) {
      const i = Math.min(play.frames.length - 1, Math.floor((t - INTRO) * play.fps));
      const frame = play.frames[i];
      return {
        pose: frame,
        face: frame.f,
        faceAlpha: 1,
        edges: play.faceEdges,
        restAlpha: 0,
        spec,
      };
    }
    const u = smooth((t - INTRO - playSpan) / INTRO);
    return {
      pose: {
        p: lerpList(last.p, idle.p, u),
        l: lerpList(last.l, idle.l, u),
        r: lerpList(last.r, idle.r, u),
      },
      face: last.f,
      faceAlpha: 1 - u,
      edges: play.faceEdges,
      restAlpha: u,
      spec,
    };
  }

  function fit(spec, x, y, w, h) {
    const scale = Math.min(w / spec.w, h / spec.h);
    const dw = spec.w * scale;
    const dh = spec.h * scale;
    return { ox: x + (w - dw) / 2, oy: y + (h - dh) / 2, scale, lw: Math.max(1.5, 2.2 * scale) };
  }

  function poly(ctx, pts, edges, map, color, lw, dots) {
    if (!pts || !pts.length) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let e = 0; e < edges.length; e++) {
      const a = edges[e][0] * 2;
      const b = edges[e][1] * 2;
      const ax = pts[a];
      const ay = pts[a + 1];
      const bx = pts[b];
      const by = pts[b + 1];
      if (!(ax >= 0) || !(ay >= 0) || !(bx >= 0) || !(by >= 0)) continue;
      const pa = map(ax, ay);
      const pb = map(bx, by);
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();
    if (!dots) return;
    const r = Math.max(1.2, lw * 0.85);
    for (let i = 0; i < pts.length; i += 2) {
      if (!(pts[i] >= 0) || !(pts[i + 1] >= 0)) continue;
      const p = map(pts[i], pts[i + 1]);
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawHand(ctx, pts, map, lw) {
    if (!pts || pts.length < 42) return;
    for (let f = 0; f < FINGER_EDGES.length; f++) {
      poly(ctx, pts, FINGER_EDGES[f], map, FINGER[f], lw, false);
      ctx.fillStyle = FINGER[f];
      const r = Math.max(1.1, lw * 0.7);
      const ids = FINGER_POINTS[f];
      for (let k = 0; k < ids.length; k++) {
        const i = ids[k] * 2;
        if (!(pts[i] >= 0) || !(pts[i + 1] >= 0)) continue;
        const p = map(pts[i], pts[i + 1]);
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawRestFace(ctx, spec, map, lw, alpha) {
    if (!alpha || alpha < 0.02) return;
    const shoulder = (spec && spec.shoulder) || 250;
    const cx = ((spec && spec.w) || 720) / 2;
    const cy = ((spec && spec.h) || 780) * ((spec && spec.shoulder_y) || 0.55);
    const hw = shoulder * 0.22;
    const hh = hw * 1.28;
    const top = cy - shoulder * 0.14 - hh * 2;
    const at = (lx, ly) => map(cx + lx * hw, top + ly * hh * 2);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = BODY;
    ctx.lineWidth = Math.max(1.25, lw * 0.95);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const center = at(0, 0.5);
    const rx = Math.abs(at(1, 0.5)[0] - center[0]);
    const ry = Math.abs(at(0, 1)[1] - at(0, 0)[1]) / 2;
    ctx.beginPath();
    ctx.ellipse(center[0], center[1], rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    const stroke = (pts) => {
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const p = at(pts[i][0], pts[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    };
    stroke([[-0.62, 0.32], [-0.28, 0.28]]);
    stroke([[0.28, 0.28], [0.62, 0.32]]);
    [-0.4, 0.4].forEach((ex) => {
      const p = at(ex, 0.42);
      ctx.beginPath();
      ctx.ellipse(p[0], p[1], rx * 0.13, ry * 0.075, 0, 0, Math.PI * 2);
      ctx.stroke();
    });
    stroke([[0, 0.48], [0, 0.64], [0.16, 0.68]]);
    const a = at(-0.32, 0.76);
    const b = at(0, 0.84);
    const c = at(0.32, 0.76);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.quadraticCurveTo(b[0], b[1], c[0], c[1]);
    ctx.stroke();
    ctx.restore();
  }

  function drawHead(ctx, pose, map, lw) {
    const p = pose.p;
    if (!p || p[0] < 0 || p[2] < 0) return;
    const a = map(p[0], p[1]);
    const b = map(p[2], p[3]);
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const span = Math.hypot(b[0] - a[0], b[1] - a[1]);
    ctx.strokeStyle = BODY;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.ellipse(mid[0], mid[1] - span * 0.55, span * 0.38, span * 0.48, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  /**
   * Dibuja el esqueleto dentro del rectángulo de la mitad derecha.
   * @param {CanvasRenderingContext2D} ctx
   */
  function draw(ctx, view, x, y, w, h) {
    if (!view) return;
    const spec = view.spec || { w: 720, h: 780 };
    const box = fit(spec, x, y, w, h);
    const map = (px, py) => [box.ox + px * box.scale, box.oy + py * box.scale];
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.fillStyle = "#121216";
    ctx.fillRect(x, y, w, h);
    const pose = view.pose || {};
    if ((view.restAlpha || 0) < 0.35 && (!view.face || view.faceAlpha < 0.35)) {
      drawHead(ctx, pose, map, box.lw);
    }
    poly(ctx, pose.p, POSE_EDGES, map, BODY, box.lw, true);
    drawRestFace(ctx, spec, map, box.lw, view.restAlpha || 0);
    if (view.face && view.faceAlpha > 0.02 && view.edges) {
      ctx.globalAlpha = view.faceAlpha;
      poly(ctx, view.face, view.edges, map, SIGN_FACE, Math.max(1, box.lw * 0.65), false);
      ctx.globalAlpha = 1;
    }
    drawHand(ctx, pose.l, map, box.lw);
    drawHand(ctx, pose.r, map, box.lw);
    ctx.restore();
  }

  window.LsaAvatar = { start, sample, draw, durationMs };
})();
