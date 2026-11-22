"""Checks of peak_check.py on hand-made numbers and tensors.

python -m pytest tasks/unlearning/test_peak_check.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import peak_check  # noqa: E402


def test_ms_term_is_one_at_the_gold_level_and_falls_linearly():
    assert peak_check.ms_term(4.0, 4.0, 4.0, 4.0) == 1.0
    assert peak_check.ms_term(5.0, 4.0, 4.0, 4.0) == pytest.approx(1 - 1 / 8)
    assert peak_check.ms_term(0.0, 0.0, 4.0, 4.0) == 0.0
    assert peak_check.ms_term(20.0, 20.0, 4.0, 4.0) == 0.0


def test_pred_term_is_one_on_the_reference_and_zero_on_the_original():
    torch.manual_seed(0)
    orig = torch.randn(50, 10) * 3
    ref = orig + torch.randn(50, 10) * 2
    assert peak_check.pred_term(ref, ref, orig) == pytest.approx(1.0, abs=1e-6)
    assert peak_check.pred_term(ref, orig, orig) == pytest.approx(0.0, abs=1e-6)


def _feat(mean: float, std: float, pred: float = 0.5, retain: float = 0.9) -> dict:
    return {"mean": mean, "std": std, "retain": retain, "pred": {"r": pred}}


def test_calibration_keeps_only_gold_levels_that_reproduce_the_peak():
    feats = {("a", 2.0): _feat(3.8, 3.8), ("a", 2.5): _feat(4.7, 4.7), ("b", 1.0): _feat(5.3, 4.9)}
    observed = (("a", 2.0, 0.28), ("a", 2.5, 0.22))
    golds = ((3.8, 3.8), (6.0, 6.0))
    kept, table = peak_check.calibrated(feats, observed, golds, ("r",), tolerance=0.02)
    assert [g for g, _, _ in kept] == [(3.8, 3.8)]
    assert table[("a", 2.0)][0] == pytest.approx(0.28)
    assert table[("a", 2.5)][0] == pytest.approx(0.22, abs=0.02)
    assert table[("b", 1.0)][0] < 0.22
