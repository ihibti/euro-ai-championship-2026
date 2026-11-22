"""Checks of slot_gain.py on hand-made draws.

python -m pytest tasks/unlearning/test_slot_gain.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import slot_gain as sg  # noqa: E402


def test_anchored_puts_the_reference_at_the_best_score_in_every_draw():
    values = np.array([[0.2, 0.1, 0.22], [0.1, 0.05, 0.09]])
    scores = sg.anchored(values, 0, 0.3)
    assert np.allclose(scores[:, 0], 0.3)
    assert scores[0, 2] == pytest.approx(0.33)
    assert scores[1, 2] == pytest.approx(0.27)


def test_joint_gain_takes_the_best_line_per_draw():
    scores = np.array([[0.31, 0.29], [0.28, 0.32], [0.27, 0.26]])
    assert sg.joint_gain(scores, 0.3) == pytest.approx((0.01 + 0.02 + 0.0) / 3)
    assert sg.joint_gain(scores[:, :1], 0.3) == pytest.approx(0.01 / 3)


def test_conditional_gains_measure_each_line_over_the_first_lines_score():
    first = np.array([0.28, 0.29, 0.30, 0.31])
    scores = np.array([[0.29], [0.28], [0.32], [0.30]])
    rows = sg.conditional_gains(scores, first, 0.285, bands=(0.0, 0.295, 1.0))
    low, high, share, beats, gains = rows[0]
    assert (low, high, share) == (0.0, 0.295, 0.5)
    assert beats[0] == pytest.approx(0.5)
    assert gains[0] == pytest.approx(0.005 / 2)
    _, _, share, beats, gains = rows[1]
    assert share == 0.5
    assert beats[0] == pytest.approx(0.5)
    assert gains[0] == pytest.approx(0.02 / 2)
