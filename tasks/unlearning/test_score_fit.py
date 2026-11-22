"""Checks of score_fit.py against the scorer's own functions, on hand-made logits.

python -m pytest tasks/unlearning/test_score_fit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import score_fit as sf  # noqa: E402
import scoring  # noqa: E402


def _logits(seed: int, rows: int = 40, classes: int = 7, scale: float = 3.0) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=(rows, classes)) * scale).astype(np.float32)


def test_parse_line_reads_factor_and_keeps_names_with_x():
    known = {"k8_e3t2_rm1_lr1e4", "x4_e20", "k11_b_soup3", "x3_ext_m80"}
    assert sf.parse_line("k8_e3t2_rm1_lr1e4x2.5", known) == ("k8_e3t2_rm1_lr1e4", 2.5)
    assert sf.parse_line("k11_b_soup3x2", known) == ("k11_b_soup3", 2.0)
    assert sf.parse_line("x4_e20", known) == ("x4_e20", 1.0)
    assert sf.parse_line("x3_ext_m80", known) == ("x3_ext_m80", 1.0)
    assert sf.parse_line("dial15", known) is None


def test_ms_table_matches_the_scorer():
    stats = np.array([[3.8, 3.7], [1.0, 2.5]])
    orig = (0.01, 0.05)
    table = sf.ms_table(stats, orig)
    for i, row in enumerate(stats):
        for m in (3, 20, 40):
            for d in (0, 17, 44):
                expected = scoring.mean_std_score(tuple(row), (sf.MEANS[m], sf.STDS[d]), orig)
                assert table[i, m, d] == pytest.approx(expected)


def test_prediction_column_matches_the_scorer():
    orig = _logits(0, scale=6.0)
    golds = [_logits(1), _logits(2, scale=1.0)]
    sub = _logits(3)
    column = sf.prediction_column(sf.gold_table(golds, orig), sub)
    for g, gold in enumerate(golds):
        gold_t, sub_t, orig_t = (torch.from_numpy(x).double() for x in (gold, sub, orig))
        expected = scoring.prediction_score(
            float(scoring.kl_rows(gold_t, sub_t).mean()),
            float(scoring.kl_rows(gold_t, orig_t).mean()),
        )
        assert column[g] == pytest.approx(expected, abs=1e-4)


def test_blend_spec_weights_probes_and_keeps_a_single_members_retain():
    specs = {
        "a": {"flip": 0.06, "spill": {"held_kl": 1.0, "spare_p_lab": 0.6}},
        "b": {"flip": 0.04, "spill": {"held_kl": 0.5, "spare_p_lab": 0.8}},
        "c": {"flip": 0.01, "retain": 0.9},
    }
    spill, flip, fixed = sf.blend_spec((("a", 3.0), ("b", 1.0)), specs)
    assert spill["held_kl"] == pytest.approx(0.875)
    assert flip == pytest.approx(0.055)
    assert fixed is None
    assert sf.blend_spec((("c", 1.0),), specs) == (None, 0.01, 0.9)


def test_fit_recovers_the_hypothesis_that_made_the_scores():
    rng = np.random.default_rng(5)
    lines = 9
    stats = np.column_stack([rng.uniform(1.0, 5.0, lines), rng.uniform(1.0, 5.0, lines)])
    ms = sf.ms_table(stats, (0.0, 0.0))
    pred = rng.uniform(0.3, 0.9, size=(4, lines))
    retain = rng.uniform(0.5, 0.8, lines)
    truth = (2, 24, 12, 18)
    s, m, d, c = truth
    observed = sf.LEVELS[c] * ms[:, m, d] * pred[s] * retain
    weights = sf.normalise(sf.log_likelihood(pred, ms, retain, observed, sigma=0.02))
    assert np.unravel_index(int(np.argmax(weights)), weights.shape) == truth
    draws = sf.draw(weights, 500)
    values = sf.predicted(draws, pred, ms, retain)
    assert values.mean(0) == pytest.approx(observed, rel=0.05)
