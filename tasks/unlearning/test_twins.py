"""Checks of twins.py on hand-made logits.

python -m pytest tasks/unlearning/test_twins.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import twins  # noqa: E402


def test_identical_twins_have_a_mixture_equal_to_their_average():
    z = np.random.default_rng(0).normal(size=(20, 5)) * 3
    mix, avg = twins.mixture_and_average([z, z, z])
    assert np.allclose(mix, avg)
    assert twins.kl_rows(mix, avg).max() == pytest.approx(0.0, abs=1e-12)


def test_disagreeing_twins_have_a_softer_mixture_than_average():
    a = np.array([[8.0, 0.0, 0.0]])
    b = np.array([[0.0, 8.0, 0.0]])
    mix, avg = twins.mixture_and_average([a, b])
    assert np.exp(mix).max() == pytest.approx(0.5, abs=0.02)
    assert twins.kl_rows(mix, avg)[0] > 0


def test_loss_stats_use_the_given_labels():
    z = np.log(np.array([[0.5, 0.5], [0.25, 0.75]]))
    losses = -np.log(np.array([0.5, 0.75]))
    expected = f"{losses.mean():.2f}/{losses.std(ddof=1):.2f}"
    assert twins.loss_stats(z, np.array([0, 1])) == expected
