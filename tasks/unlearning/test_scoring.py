"""Checks of scoring.py against cases with a known answer.

python -m pytest tasks/unlearning/test_scoring.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import scoring  # noqa: E402


def _case(seed: int = 0, n: int = 200, classes: int = 10):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=generator)
    orig = torch.randn(n, classes, generator=generator)
    orig[torch.arange(n), labels] += 12.0
    gold = torch.randn(n, classes, generator=generator) * 2.0
    gold[torch.arange(n), labels] += 2.0
    return orig, gold, labels


def test_submission_equal_to_original_scores_zero():
    orig, gold, labels = _case()
    terms = scoring.forget_terms(orig, gold, orig, labels)
    assert terms == (0.0, 0.0)
    assert scoring.forget_score([terms]) == 0.0


def test_submission_equal_to_gold_scores_one():
    orig, gold, labels = _case()
    terms = scoring.forget_terms(gold, gold, orig, labels)
    assert terms[0] == pytest.approx(1.0)
    assert terms[1] == pytest.approx(1.0)


def test_terms_are_averaged_over_groups_before_the_product():
    assert scoring.forget_score([(1.0, 1.0), (0.0, 0.0)]) == pytest.approx(0.25)


def test_mean_std_score_clips_an_overshoot_to_zero():
    assert scoring.mean_std_score((10.0, 0.0), (1.0, 1.0), (0.0, 0.0)) == 0.0
    assert scoring.mean_std_score((0.5, 0.5), (1.0, 1.0), (0.0, 0.0)) == pytest.approx(0.5)


def test_prediction_score_is_one_minus_the_kl_ratio():
    assert scoring.prediction_score(0.25, 1.0) == pytest.approx(0.75)
    assert scoring.prediction_score(2.0, 1.0) == 0.0


def test_measured_loss_statistics_override_the_proxy():
    orig, gold, labels = _case()
    gold_stats = scoring.loss_stats(gold, labels)
    terms = scoring.forget_terms(orig, gold, orig, labels, sub_stats=gold_stats)
    assert terms[0] == pytest.approx(1.0)
    assert terms[1] == 0.0


def test_retain_is_one_for_the_original_and_drops_with_scaling():
    orig, _, labels = _case()
    assert scoring.retain_score(orig, orig, labels) == pytest.approx(1.0)
    scaled = scoring.retain_score(orig / 4, orig, labels)
    assert 0.0 < scaled < 1.0


def test_scaling_keeps_every_prediction():
    orig, _, _ = _case()
    assert torch.equal((orig / 4).argmax(1), orig.argmax(1))
