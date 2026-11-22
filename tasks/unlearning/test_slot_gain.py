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


def test_conditional_gains_skip_bands_without_draws():
    first = np.full(3, 0.29)
    rows = sg.conditional_gains(np.zeros((3, 1)), first, 0.29, bands=(0.0, 0.28, 0.3, 1.0))
    assert [(low, high) for low, high, *_ in rows] == [(0.28, 0.3)]


def test_overlap_is_the_shared_normalised_weight():
    assert sg.overlap({"a": 0.5, "b": 0.5}, {"a": 1.0}) == pytest.approx(0.5)
    assert sg.overlap({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert sg.overlap({"a": 1.0}, {"b": 1.0}) == 0.0


def test_shared_anchor_with_one_anchor_is_the_best_line_anchor():
    model = np.array([0.2, 0.3])
    anchor = np.array([[0.25], [0.2]])
    out = sg.shared_anchor(model, anchor, np.array([0.3]), np.array([0.7]))
    assert np.allclose(out, model * 0.3 / anchor[:, 0])


def test_shared_anchor_takes_the_weighted_geometric_mean_of_ratios():
    model = np.array([1.0])
    anchors = np.array([[1.0, 1.0]])
    out = sg.shared_anchor(model, anchors, np.array([1.21, 1.0]), np.array([1.0, 1.0]))
    assert out[0] == pytest.approx(1.1)


def test_shared_scores_anchor_on_shared_runs_and_fall_back_to_the_best_line():
    names = ["k11_b_s1x2", "k8_e3t2_rm1_lr1e4x2", "k11_b_s2x2", "x4_e20x2"]
    values = np.array([[0.2, 0.25, 0.22, 0.1], [0.3, 0.2, 0.33, 0.1]])
    inputs = {
        "candidates": {"k11_b_s1": {}, "k11_b_s2": {}, "k8_e3t2_rm1_lr1e4": {}, "x4_e20": {}},
        "observed": {"k11_b_s1x2": 0.28, "k8_e3t2_rm1_lr1e4x2": 0.29},
    }
    out = sg.shared_scores(values, names, inputs, reference=1, best=0.29)
    assert np.allclose(out[:, 2], 0.29 * values[:, 2] / values[:, 1])
    assert np.allclose(out[:, 3], 0.29 * values[:, 3] / values[:, 1])
    soup = ["k11_b_soup3x2"]
    out = sg.shared_scores(
        np.column_stack([values[:, :2], values[:, 2]]), names[:2] + soup, inputs, 1, 0.29
    )
    w = np.array([1 / 3, 1 / 3])
    ratios = np.log(np.array([0.28, 0.29])[None, :] / values[:, :2])
    assert np.allclose(out[:, 2], values[:, 2] * np.exp(ratios @ (w / w.sum())))
