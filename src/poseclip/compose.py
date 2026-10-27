"""Une clips `.pose` en una frase: recorte, park, EMA, canvas y puentes.

No escribe mp4. El navegador dibuja los frames con `canvas.captureStream`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.ma as ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions

WRIST_LEFT_IDX = 15
WRIST_RIGHT_IDX = 16
LEFT_ELBOW_IDX = 13
RIGHT_ELBOW_IDX = 14
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_PINKY_IDX = 17
RIGHT_PINKY_IDX = 18
LEFT_INDEX_TIP_IDX = 19
RIGHT_INDEX_TIP_IDX = 20
LEFT_THUMB_IDX = 21
RIGHT_THUMB_IDX = 22
HAND_WRIST = 0
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_PINKY_TIP = 20
LEG_BELOW_HIP = slice(25, 33)
POSE_POINTS = 33
FACE_POINTS = 468
HAND_POINTS = 21
LEFT_HAND_START = POSE_POINTS + FACE_POINTS
RIGHT_HAND_START = LEFT_HAND_START + HAND_POINTS

DEFAULT_FPS = 30.0
DEFAULT_EMA = 0.50
DEFAULT_TRIM = 0.25
BRIDGE_MIN_FRAMES = 4
BRIDGE_MAX_FRAMES = 8
BRIDGE_DIST_SHORT_PX = 40.0
BRIDGE_DIST_LONG_PX = 180.0
BRIDGE_EASING = "smoothstep"
BRIDGE_FIXED_FRAMES = None
FILL_MISSING = True
MISSING_CONF_MIN = 0.30
MISSING_XY_PX = 8.0
FILL_MAX_GAP = 8
PARK_IDLE_HANDS = True
IDLE_HAND_HIT = 0.10
DUPLICATE_HAND_FRAC = 0.07
CANVAS_WIDTH = 720
CANVAS_HEIGHT = 780
CANVAS_SHOULDER_PX = 250.0
CANVAS_SHOULDER_Y = 0.55
CENTER_ON_SHOULDERS = True
HIDE_LEGS = True


@dataclass(frozen=True)
class ComposeSettings:
    fps: float = DEFAULT_FPS
    trim: float = DEFAULT_TRIM
    missing_conf: float = MISSING_CONF_MIN
    missing_xy: float = MISSING_XY_PX
    fill_missing: bool = FILL_MISSING
    fill_max_gap: int = FILL_MAX_GAP
    park_idle: bool = PARK_IDLE_HANDS
    ema: float = DEFAULT_EMA
    center: bool = CENTER_ON_SHOULDERS
    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    shoulder_px: float = CANVAS_SHOULDER_PX
    shoulder_y: float = CANVAS_SHOULDER_Y
    hide_legs: bool = HIDE_LEGS
    bridge_min: int = BRIDGE_MIN_FRAMES
    bridge_max: int = BRIDGE_MAX_FRAMES
    bridge_dist_short: float = BRIDGE_DIST_SHORT_PX
    bridge_dist_long: float = BRIDGE_DIST_LONG_PX
    easing: str = BRIDGE_EASING
    bridge_fixed: int | None = BRIDGE_FIXED_FRAMES


SETTINGS = ComposeSettings()

# Mano en reposo, dedos hacia +Y (abajo en imagen). Pulgar hacia −X.
_REST_HAND_XY = np.array(
    [
        [0.00, 0.00],
        [-0.18, 0.10],
        [-0.24, 0.22],
        [-0.26, 0.32],
        [-0.28, 0.40],
        [-0.10, 0.14],
        [-0.10, 0.32],
        [-0.10, 0.44],
        [-0.10, 0.54],
        [0.00, 0.14],
        [0.00, 0.34],
        [0.00, 0.48],
        [0.00, 0.58],
        [0.10, 0.14],
        [0.10, 0.32],
        [0.10, 0.44],
        [0.10, 0.52],
        [0.18, 0.12],
        [0.20, 0.28],
        [0.20, 0.38],
        [0.20, 0.46],
    ],
    dtype=np.float32,
)


def load_pose(path: Path) -> Pose:
    pose = Pose.read(path.read_bytes())
    return pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    )


def interpolate(pose: Pose, fps: float) -> Pose:
    if abs(float(pose.body.fps) - fps) < 0.5:
        return pose
    pose.body = pose.body.interpolate(new_fps=fps, kind="linear")
    return pose


def _valid_mask(data: np.ndarray, conf: np.ndarray, conf_min: float, xy_min: float) -> np.ndarray:
    xy = np.linalg.norm(data[:, 0, :, :2], axis=-1)
    return (conf[:, 0, :] >= conf_min) & (xy >= xy_min)


def _fill_series(values: np.ndarray, ok: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    out = values.copy()
    filled = 0
    length = len(ok)
    if not ok.any() or ok.all():
        return out, 0
    index = 0
    while index < length:
        if ok[index]:
            index += 1
            continue
        start = index
        while index < length and not ok[index]:
            index += 1
        end = index
        left = start - 1
        right = end if end < length else -1
        gap = end - start
        if left >= 0 and right >= 0 and gap <= max_gap:
            xs = np.array([left, right], dtype=np.float32)
            out[start:end] = np.interp(
                np.arange(start, end, dtype=np.float32), xs, values[[left, right]]
            )
            filled += gap
        elif left >= 0:
            out[start:end] = values[left]
            filled += gap
        elif right >= 0:
            out[start:end] = values[right]
            filled += gap
    return out, filled


def fill_missing_landmarks(
    pose: Pose,
    conf_min: float,
    xy_min: float,
    max_gap: int = FILL_MAX_GAP,
) -> tuple[Pose, int]:
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    valid = _valid_mask(data, conf, conf_min, xy_min)
    _frames, _people, points, dims = data.shape
    filled = 0
    for point in range(points):
        ok = valid[:, point]
        if ok.all() or not ok.any():
            continue
        conf[:, 0, point], _ = _fill_series(conf[:, 0, point], ok, max_gap)
        for axis in range(dims):
            data[:, 0, point, axis], n_fill = _fill_series(data[:, 0, point, axis], ok, max_gap)
            filled += n_fill
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    return pose, filled


def _hand_frame_hit(valid: np.ndarray, start: int) -> float:
    return float(valid[:, start : start + HAND_POINTS].any(axis=1).mean())


def _hand_centroid(data: np.ndarray, start: int) -> np.ndarray:
    return data[:, 0, start : start + HAND_POINTS, :2].mean(axis=1)


def _shoulder_width_px(data: np.ndarray, conf: np.ndarray) -> float:
    widths = np.linalg.norm(
        data[:, 0, LEFT_SHOULDER_IDX, :2] - data[:, 0, RIGHT_SHOULDER_IDX, :2],
        axis=-1,
    )
    ok = (conf[:, 0, LEFT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        conf[:, 0, RIGHT_SHOULDER_IDX] >= MISSING_CONF_MIN
    )
    if ok.any():
        return float(max(np.median(widths[ok]), 80.0))
    return 400.0


def detect_idle_hands(
    pose: Pose,
    conf_min: float,
    xy_min: float,
    idle_hit: float = IDLE_HAND_HIT,
    dup_frac: float = DUPLICATE_HAND_FRAC,
) -> tuple[bool, bool]:
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if data.shape[2] < RIGHT_HAND_START + HAND_POINTS:
        return False, False
    valid = _valid_mask(data, conf, conf_min, xy_min)
    left_hit = _hand_frame_hit(valid, LEFT_HAND_START)
    right_hit = _hand_frame_hit(valid, RIGHT_HAND_START)
    idle_left = left_hit < idle_hit
    idle_right = right_hit < idle_hit
    both = valid[:, LEFT_HAND_START : LEFT_HAND_START + HAND_POINTS].any(axis=1) & valid[
        :, RIGHT_HAND_START : RIGHT_HAND_START + HAND_POINTS
    ].any(axis=1)
    if both.any() and not idle_left and not idle_right:
        left_c = _hand_centroid(data, LEFT_HAND_START)
        right_c = _hand_centroid(data, RIGHT_HAND_START)
        apart = float(np.median(np.linalg.norm(left_c[both] - right_c[both], axis=-1)))
        if apart < dup_frac * _shoulder_width_px(data, conf):
            if left_hit <= right_hit:
                idle_left = True
            else:
                idle_right = True
    if idle_left and idle_right:
        if left_hit >= right_hit:
            idle_left = False
        else:
            idle_right = False
    return idle_left, idle_right


def _park_arm(data: np.ndarray, conf: np.ndarray, *, left: bool) -> None:
    if left:
        shoulder_i, elbow_i, wrist_i = LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, WRIST_LEFT_IDX
        hip_i = LEFT_HIP_IDX
        hand_start = LEFT_HAND_START
        tips = (LEFT_PINKY_IDX, LEFT_INDEX_TIP_IDX, LEFT_THUMB_IDX)
        side = 1.0
        hand_xy = _REST_HAND_XY
    else:
        shoulder_i, elbow_i, wrist_i = RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, WRIST_RIGHT_IDX
        hip_i = RIGHT_HIP_IDX
        hand_start = RIGHT_HAND_START
        tips = (RIGHT_PINKY_IDX, RIGHT_INDEX_TIP_IDX, RIGHT_THUMB_IDX)
        side = -1.0
        hand_xy = _REST_HAND_XY.copy()
        hand_xy[:, 0] *= -1.0

    width = _shoulder_width_px(data, conf)
    shoulder = data[:, 0, shoulder_i]
    hip = data[:, 0, hip_i]
    hip_ok = (conf[:, 0, hip_i] >= 0.15) & (np.linalg.norm(hip[:, :2], axis=-1) >= MISSING_XY_PX)
    down = np.zeros_like(shoulder)
    down[:, 1] = width * 1.55
    if hip_ok.any():
        estimated = hip - shoulder
        estimated[~hip_ok] = down[~hip_ok]
        down = estimated
        down[:, 1] = np.maximum(down[:, 1], width * 0.9)
    outward = np.zeros_like(shoulder)
    outward[:, 0] = side * width * 0.12
    elbow = shoulder + 0.42 * down + 0.55 * outward
    wrist = shoulder + 0.90 * down + 0.70 * outward
    data[:, 0, elbow_i] = elbow
    data[:, 0, wrist_i] = wrist
    conf[:, 0, elbow_i] = np.maximum(conf[:, 0, elbow_i], 0.55)
    conf[:, 0, wrist_i] = np.maximum(conf[:, 0, wrist_i], 0.55)

    scale = width * 0.22
    offsets = np.zeros((HAND_POINTS, data.shape[-1]), dtype=np.float32)
    offsets[:, :2] = hand_xy * scale
    data[:, 0, hand_start : hand_start + HAND_POINTS] = wrist[:, None, :] + offsets[None, :, :]
    conf[:, 0, hand_start : hand_start + HAND_POINTS] = 0.55
    tip_src = (HAND_PINKY_TIP, HAND_INDEX_TIP, HAND_THUMB_TIP)
    for pose_idx, hand_idx in zip(tips, tip_src):
        data[:, 0, pose_idx] = data[:, 0, hand_start + hand_idx]
        conf[:, 0, pose_idx] = 0.55


def snap_pose_arm_to_hand(data: np.ndarray, conf: np.ndarray, *, left: bool) -> None:
    if left:
        shoulder_i, elbow_i, wrist_i = LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, WRIST_LEFT_IDX
        hand_start = LEFT_HAND_START
        hip_i = LEFT_HIP_IDX
        tips = (LEFT_PINKY_IDX, LEFT_INDEX_TIP_IDX, LEFT_THUMB_IDX)
    else:
        shoulder_i, elbow_i, wrist_i = RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, WRIST_RIGHT_IDX
        hand_start = RIGHT_HAND_START
        hip_i = RIGHT_HIP_IDX
        tips = (RIGHT_PINKY_IDX, RIGHT_INDEX_TIP_IDX, RIGHT_THUMB_IDX)
    hand_wrist = data[:, 0, hand_start + HAND_WRIST]
    data[:, 0, wrist_i] = hand_wrist
    conf[:, 0, wrist_i] = np.maximum(conf[:, 0, wrist_i], conf[:, 0, hand_start + HAND_WRIST])
    shoulder = data[:, 0, shoulder_i]
    hip = data[:, 0, hip_i]
    mid = 0.5 * (shoulder + hand_wrist)
    data[:, 0, elbow_i] = mid + 0.16 * (hip - shoulder)
    conf[:, 0, elbow_i] = np.maximum(conf[:, 0, elbow_i], 0.5)
    tip_src = (HAND_PINKY_TIP, HAND_INDEX_TIP, HAND_THUMB_TIP)
    for pose_idx, hand_idx in zip(tips, tip_src):
        data[:, 0, pose_idx] = data[:, 0, hand_start + hand_idx]
        conf[:, 0, pose_idx] = np.maximum(conf[:, 0, pose_idx], 0.5)


def ema_body(pose: Pose, alpha: float) -> Pose:
    if alpha >= 1.0 or pose.body.data.shape[0] < 2:
        return pose
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    out = np.empty_like(data)
    out[0] = data[0]
    keep = 1.0 - alpha
    for idx in range(1, len(data)):
        out[idx] = alpha * data[idx] + keep * out[idx - 1]
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(out), conf)
    return pose


def slice_pose(pose: Pose, start: int, end: int) -> Pose:
    data = pose.body.data[start:end]
    conf = pose.body.confidence[start:end]
    return Pose(pose.header, NumPyPoseBody(pose.body.fps, data, conf))


def trim_pose(pose: Pose, fraction: float) -> Pose:
    frames = pose.body.data.shape[0]
    if fraction <= 0 or frames < 12:
        return pose
    cut = max(1, int(round(frames * fraction)))
    start, end = cut, frames - cut
    if end - start < 8:
        return pose
    return slice_pose(pose, start, end)


def place_on_canvas(
    pose: Pose,
    width: int,
    height: int,
    shoulder_px: float,
    shoulder_y: float,
    hide_legs: bool = True,
) -> Pose:
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    left = data[:, 0, LEFT_SHOULDER_IDX, :]
    right = data[:, 0, RIGHT_SHOULDER_IDX, :]
    left_ok = (conf[:, 0, LEFT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        np.linalg.norm(left[:, :2], axis=-1) >= MISSING_XY_PX
    )
    right_ok = (conf[:, 0, RIGHT_SHOULDER_IDX] >= MISSING_CONF_MIN) & (
        np.linalg.norm(right[:, :2], axis=-1) >= MISSING_XY_PX
    )
    ok = left_ok & right_ok
    mid = 0.5 * (left + right)
    widths = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
    times = np.arange(len(data), dtype=np.float32)
    if ok.any():
        for axis in range(data.shape[-1]):
            mid[:, axis] = np.interp(times, times[ok], mid[ok, axis])
        scale = float(shoulder_px) / max(float(np.median(widths[ok])), 1e-3)
    else:
        scale = 1.0
        mid[:] = 0.0
    cx = width / 2.0
    cy = height * float(shoulder_y)
    placed = data.copy()
    placed[..., 0] = (data[..., 0] - mid[:, None, None, 0]) * scale + cx
    placed[..., 1] = (data[..., 1] - mid[:, None, None, 1]) * scale + cy
    if data.shape[-1] > 2:
        placed[..., 2] = (data[..., 2] - mid[:, None, None, 2]) * scale
    if hide_legs and placed.shape[2] >= 33:
        conf = conf.copy()
        conf[:, 0, LEG_BELOW_HIP] = 0.0
    pose.body = NumPyPoseBody(pose.body.fps, ma.array(placed), conf)
    pose.header.dimensions = PoseHeaderDimensions(width, height, pose.header.dimensions.depth)
    return pose


def prepare_sign(path: Path, settings: ComposeSettings | None = None) -> Pose:
    settings = settings or SETTINGS
    pose = interpolate(load_pose(path), settings.fps)
    pose = trim_pose(pose, settings.trim)
    idle_left, idle_right = detect_idle_hands(pose, settings.missing_conf, settings.missing_xy)
    if settings.fill_missing:
        pose, _filled = fill_missing_landmarks(
            pose,
            settings.missing_conf,
            settings.missing_xy,
            max_gap=settings.fill_max_gap,
        )
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if data.shape[2] >= RIGHT_HAND_START + HAND_POINTS:
        if settings.park_idle and idle_left:
            _park_arm(data, conf, left=True)
        elif not idle_left:
            snap_pose_arm_to_hand(data, conf, left=True)
        if settings.park_idle and idle_right:
            _park_arm(data, conf, left=False)
        elif not idle_right:
            snap_pose_arm_to_hand(data, conf, left=False)
        pose.body = NumPyPoseBody(pose.body.fps, ma.array(data), conf)
    pose = ema_body(pose, settings.ema)
    if settings.center:
        pose = place_on_canvas(
            pose,
            settings.canvas_width,
            settings.canvas_height,
            settings.shoulder_px,
            settings.shoulder_y,
            hide_legs=settings.hide_legs,
        )
    return pose


def wrist_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    left = float(np.linalg.norm(frame_a[WRIST_LEFT_IDX] - frame_b[WRIST_LEFT_IDX]))
    right = float(np.linalg.norm(frame_a[WRIST_RIGHT_IDX] - frame_b[WRIST_RIGHT_IDX]))
    return 0.5 * (left + right)


def ease_weights(t: np.ndarray, easing: str) -> np.ndarray:
    if easing == "linear":
        return t
    return t * t * (3.0 - 2.0 * t)


def bridge_length(
    dist_px: float,
    min_len: int,
    max_len: int,
    dist_short: float,
    dist_long: float,
    fixed: int | None,
) -> int:
    if fixed is not None and fixed > 0:
        return int(fixed)
    span = max(dist_long - dist_short, 1e-6)
    mix = float(np.clip((dist_px - dist_short) / span, 0.0, 1.0))
    return int(round(min_len + (max_len - min_len) * mix))


def bridge_poses(
    pose_a: Pose,
    pose_b: Pose,
    settings: ComposeSettings,
) -> tuple[np.ndarray, np.ndarray]:
    last = np.ma.filled(pose_a.body.data[-1], 0.0)
    first = np.ma.filled(pose_b.body.data[0], 0.0)
    last_c = np.ma.filled(pose_a.body.confidence[-1], 0.0)
    first_c = np.ma.filled(pose_b.body.confidence[0], 0.0)
    dist = wrist_distance(last[0], first[0])
    n = bridge_length(
        dist,
        settings.bridge_min,
        settings.bridge_max,
        settings.bridge_dist_short,
        settings.bridge_dist_long,
        settings.bridge_fixed,
    )
    n = max(n, 1)
    t = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
    w = ease_weights(t, settings.easing).reshape(-1, 1, 1, 1)
    data = (1.0 - w) * last[None, ...] + w * first[None, ...]
    wc = ease_weights(t, settings.easing).reshape(-1, 1, 1)
    conf = (1.0 - wc) * last_c[None, ...] + wc * first_c[None, ...]
    return data.astype(np.float32), conf.astype(np.float32)


def compose_sequence(poses: list[Pose], settings: ComposeSettings | None = None) -> Pose:
    if not poses:
        raise ValueError("No hay clips para componer.")
    settings = settings or SETTINGS
    data_parts = [np.ma.filled(poses[0].body.data, 0.0)]
    conf_parts = [np.ma.filled(poses[0].body.confidence, 0.0)]
    for index in range(1, len(poses)):
        bridge_data, bridge_conf = bridge_poses(poses[index - 1], poses[index], settings)
        data_parts.append(bridge_data)
        conf_parts.append(bridge_conf)
        data_parts.append(np.ma.filled(poses[index].body.data, 0.0))
        conf_parts.append(np.ma.filled(poses[index].body.confidence, 0.0))
    body = NumPyPoseBody(
        poses[0].body.fps,
        ma.array(np.concatenate(data_parts, axis=0)),
        np.concatenate(conf_parts, axis=0),
    )
    return Pose(poses[0].header, body)
