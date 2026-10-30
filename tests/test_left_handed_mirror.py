"""Left-handed mirror must relabel pose L/R, not only flip X and swap hands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.landmarks import mirror_landmarks_for_left_handed  # noqa: E402

POSE_DIM = 33 * 3
HAND_DIM = 21 * 3


def _xyz(vec: np.ndarray, index: int) -> np.ndarray:
    return vec[index * 3 : index * 3 + 3]


def test_pose_wrists_follow_swapped_hands():
    vec = np.zeros(POSE_DIM + HAND_DIM * 2, dtype=np.float32)
    vec[11 * 3] = 0.50
    vec[12 * 3] = -0.30
    vec[15 * 3] = 0.80
    vec[16 * 3] = -0.10
    vec[POSE_DIM] = 0.80
    vec[POSE_DIM + HAND_DIM] = -0.10

    out = mirror_landmarks_for_left_handed(vec)

    np.testing.assert_allclose(_xyz(out, 11), [0.30, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_xyz(out, 12), [-0.50, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_xyz(out, 15), [0.10, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_xyz(out, 16), [-0.80, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[POSE_DIM : POSE_DIM + 3], [0.10, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        out[POSE_DIM + HAND_DIM : POSE_DIM + HAND_DIM + 3],
        [-0.80, 0.0, 0.0],
        atol=1e-6,
    )


def test_right_hand_slot_holds_mirrored_dominant_left_hand():
    vec = np.zeros(POSE_DIM + HAND_DIM * 2, dtype=np.float32)
    for i in range(21):
        vec[POSE_DIM + i * 3] = 0.2 + i * 0.01
        vec[POSE_DIM + i * 3 + 1] = 0.4
    out = mirror_landmarks_for_left_handed(vec)
    for i in range(21):
        assert out[POSE_DIM + i * 3] == 0.0
        np.testing.assert_allclose(
            out[POSE_DIM + HAND_DIM + i * 3],
            -(0.2 + i * 0.01),
            atol=1e-6,
        )


def test_idempotent_double_mirror():
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(POSE_DIM + HAND_DIM * 2).astype(np.float32)
    twice = mirror_landmarks_for_left_handed(mirror_landmarks_for_left_handed(vec))
    np.testing.assert_allclose(twice, vec, atol=1e-5)


if __name__ == "__main__":
    test_pose_wrists_follow_swapped_hands()
    print("ok test_pose_wrists_follow_swapped_hands")
    test_right_hand_slot_holds_mirrored_dominant_left_hand()
    print("ok test_right_hand_slot_holds_mirrored_dominant_left_hand")
    test_idempotent_double_mirror()
    print("ok test_idempotent_double_mirror")
    print("all left-handed mirror tests passed")
