"""Checks of morning_screen.py on hand-made numbers.

python -m pytest tasks/unlearning/test_morning_screen.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import morning_screen as ms  # noqa: E402


def _specs() -> dict[str, dict]:
    return {
        "a": {"spill": {"spare_p_lab": 0.6, "held_kl": 1.0, "spare_kl": 0.6}, "flip": 0.06},
        "b": {"spill": {"spare_p_lab": 0.8, "held_kl": 0.5, "spare_kl": 0.2}, "flip": 0.04},
    }


def test_blend_weights_logits_and_probe_statistics():
    logits = {"a": np.zeros((3, 4)), "b": np.ones((3, 4))}
    z, spill, flip = ms.blend((("a", 3.0), ("b", 1.0)), logits, _specs())
    assert np.allclose(z, 0.25)
    assert spill["held_kl"] == pytest.approx(0.875)
    assert spill["spare_p_lab"] == pytest.approx(0.65)
    assert flip == pytest.approx(0.055)


def test_operating_point_ratio_is_one_where_both_factors_agree():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(300, 6))
    z[:, 0] -= 2.0
    labels = np.zeros(300, dtype=int)
    loss_factor = ms.factor_for_loss(z, labels, ms.LEVEL)
    point = ms.operating_point(z, labels, loss_factor / ms.RATIO)
    assert point["ratio"] == pytest.approx(1.0)
    assert point["loss"][0] == pytest.approx(ms.LEVEL, abs=1e-6)
    assert ms.operating_point(z, labels, 2 * loss_factor / ms.RATIO)["ratio"] == pytest.approx(0.5)
