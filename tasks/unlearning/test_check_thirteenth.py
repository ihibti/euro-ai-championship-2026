"""Checks of check_thirteenth.py on hand-made numbers.

python -m pytest tasks/unlearning/test_check_thirteenth.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_thirteenth as ct  # noqa: E402


def test_class_disagreement_counts_rows_with_another_top_class():
    a = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    b = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    assert ct.class_disagreement(a, b) == pytest.approx(0.5)


def test_kl_is_zero_for_equal_logits_and_positive_otherwise():
    z = np.random.default_rng(0).normal(size=(10, 4)) * 3
    assert ct.kl_t2(z, z) == pytest.approx(0.0, abs=1e-12)
    assert ct.kl_t2(z, z[::-1]) > 0


def test_all_four_checks_must_hold():
    good = ct.checks(0.08, 0.05, (3.7, 3.9), 0.055)
    assert all(good.values())
    assert not ct.checks(0.04, 0.05, (3.7, 3.9), 0.055)["cross-fits disagree"]
    assert not ct.checks(0.08, 0.02, (3.7, 3.9), 0.055)["differs from the 0.2858"]
    assert not ct.checks(0.08, 0.05, (4.3, 3.9), 0.055)["loss near 3.8/3.8"]
    assert not ct.checks(0.08, 0.05, (3.7, 3.9), 0.065)["probe flip at most 6%"]


def test_factor_for_loss_reaches_the_level():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(200, 5))
    z[:, 0] -= 2.0
    labels = np.zeros(200, dtype=int)
    factor = ct.factor_for_loss(z, labels, 3.0)
    assert ct.loss_stats(z * factor, labels)[0] == pytest.approx(3.0, abs=1e-6)
    assert ct.factor_for_loss(z, labels, 2.0) < factor


def test_target_losses_without_a_step_are_the_original_loss():
    orig = np.random.default_rng(2).normal(size=(50, 6)) * 4
    columns = np.arange(6)
    single, mixture = ct.target_losses(orig, [orig, orig], columns)
    expected = ct.loss_stats(orig, orig.argmax(1))
    assert single == pytest.approx(expected)
    assert mixture == pytest.approx(expected)


def test_target_losses_mixture_of_equal_steps_is_the_single_target():
    rng = np.random.default_rng(3)
    orig = rng.normal(size=(50, 6)) * 4
    pred = orig + rng.normal(size=(50, 6))
    single, mixture = ct.target_losses(orig, [pred, pred, pred], np.arange(6))
    assert mixture == pytest.approx(single)
