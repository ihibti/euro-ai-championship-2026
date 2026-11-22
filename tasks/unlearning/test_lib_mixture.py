"""Checks of lib.mixture_targets on hand-made logits.

python -m pytest tasks/unlearning/test_lib_mixture.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import lib  # noqa: E402

CLASSES = 6


def _inputs(seed: int = 0, batch: int = 8):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, CLASSES, generator=g) * 3
    labels = torch.randint(0, CLASSES, (batch,), generator=g)
    zero = torch.zeros(CLASSES)
    return g, logits, labels, zero


def test_one_step_gives_the_soft_target_of_that_step():
    g, logits, labels, zero = _inputs()
    step = torch.randn(8, CLASSES, generator=g)
    mixed = lib.mixture_targets(logits, step.unsqueeze(0), labels, zero, zero, tau=2.0)
    single = lib.soft_targets(logits + step, labels, zero, zero, tau=2.0)
    assert torch.allclose(mixed, single, atol=1e-6)


def test_mixture_is_the_mean_of_the_probabilities():
    g, logits, labels, zero = _inputs(1)
    steps = torch.randn(3, 8, CLASSES, generator=g) * 2
    mixed = lib.mixture_targets(logits, steps, labels, zero, zero, tau=2.0)
    probs = torch.stack(
        [lib.soft_targets(logits + s, labels, zero, zero, tau=2.0).exp() for s in steps]
    ).mean(0)
    assert torch.allclose(mixed.exp(), probs, atol=1e-6)
    assert mixed.exp().sum(1) == pytest.approx(torch.ones(8), abs=1e-5)


def test_disagreeing_steps_spread_the_target():
    _, logits, labels, zero = _inputs(2, batch=1)
    logits = torch.zeros(1, CLASSES)
    up = torch.zeros(2, 1, CLASSES)
    up[0, 0, 1] = 10.0
    up[1, 0, 2] = 10.0
    mixed = lib.mixture_targets(logits, up, labels, zero, zero, tau=1.0).exp()[0]
    assert float(mixed[1]) == pytest.approx(float(mixed[2]), abs=1e-6)
    assert float(mixed[1]) == pytest.approx(0.5, abs=0.01)
