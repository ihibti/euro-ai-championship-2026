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


def test_retain_modes_keep_flatten_or_halve_the_differences():
    feats = {("a", 2.0): _feat(3.8, 3.8, retain=0.7), ("b", 1.0): _feat(3.8, 3.8, retain=0.8)}
    anchor = ("a", 2.0)
    assert peak_check.with_retain(feats, "probe", anchor) is feats
    assert peak_check.with_retain(feats, "flat", anchor)[("b", 1.0)]["retain"] == pytest.approx(0.7)
    assert peak_check.with_retain(feats, "half", anchor)[("b", 1.0)]["retain"] == pytest.approx(
        0.75
    )
    assert feats[("b", 1.0)]["retain"] == 0.8


def test_observed_scores_parse_with_the_best_first():
    rows = peak_check.parse_observed(["a_b:2:0.2222", "a_b:2.5:0.2837", "c:1.25:0.25"])
    assert rows[0] == ("a_b", 2.5, 0.2837)
    assert [r[2] for r in rows] == [0.2837, 0.25, 0.2222]


def test_worst_case_skips_readings_that_kept_nothing():
    tables = {
        "probe": {},
        "half": {("a", 2.0): (0.29, 0.28, 0.30)},
        "flat": {("a", 2.0): (0.27, 0.26, 0.28)},
    }
    live = [m for m in ("probe", "half", "flat") if tables[m]]
    assert peak_check.worst_case(tables, live, ("a", 2.0)) == 0.27
    assert peak_check.worst_case(tables, live, ("b", 1.0)) == float("-inf")
