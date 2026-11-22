"""Checks of the ledger's pure helpers on synthetic logits.

python -m pytest tasks/unlearning/test_ledger.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import ledger  # noqa: E402
import scoring  # noqa: E402


def _orig(n: int = 300, classes: int = 20, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=generator)
    logits = torch.randn(n, classes, generator=generator)
    logits[torch.arange(n), labels] += 10.0
    return logits, labels


def test_gate_pass_is_the_share_of_the_band_that_passes():
    assert ledger.gate_pass(0.0, 72.0, 80.0) == 1.0
    assert ledger.gate_pass(0.5, 72.0, 80.0) == 0.0
    assert ledger.gate_pass(0.1, 70.0, 80.0) == pytest.approx((80.0 - 70.0 / 0.9) / 10.0)


def test_within_class_permutation_stays_inside_each_class():
    _, labels = _orig()
    perm = ledger.within_class_permutation(labels)
    assert torch.equal(labels[perm], labels)
    assert sorted(perm.tolist()) == list(range(len(labels)))


def test_effective_temperature_inverts_a_known_temperature():
    orig, labels = _orig()
    target = float(scoring.label_prob_t2(orig / 2.5, labels).mean())
    assert ledger.effective_temperature(orig, labels, target) == pytest.approx(2.5, abs=0.05)


def _flipped(orig: torch.Tensor, labels: torch.Tensor, share: float, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    sub = orig / 2.5
    rows = torch.rand(len(labels), generator=generator) < share
    wrong = (labels + 1) % orig.shape[1]
    sub[rows, labels[rows]] -= 6.0
    sub[rows, wrong[rows]] += 6.0
    return sub


def test_sharpening_back_by_the_temperature_removes_a_pure_temperature_spill():
    orig, _ = _orig()
    tempered = orig / 2.5
    kl = float(scoring.kl_rows(orig, tempered).mean())
    spill = {"held_kl": kl, "held_acc": 1.0, "spare_kl": kl, "spare_acc": 1.0}
    unsharpened = 1.0 / (1.0 + kl)
    assert ledger.sharpened_retain(spill, 2.5, orig, tempered) == pytest.approx(1.0, abs=1e-4)
    assert ledger.sharpened_retain(spill, 1.5, orig, tempered) > unsharpened


def test_a_factor_of_one_keeps_the_measured_retain():
    orig, labels = _orig()
    sub = _flipped(orig, labels, 0.3)
    spill = {"held_kl": 1.1, "held_acc": 0.87, "spare_kl": 0.78, "spare_acc": 0.987}
    measured = (0.45 * 0.87 + 0.55 * 0.987) / (1.0 + 0.45 * 1.1 + 0.55 * 0.78)
    assert ledger.sharpened_retain(spill, 1.0, orig, sub) == pytest.approx(measured)


def test_changed_predictions_keep_their_drift_when_sharpened():
    orig, labels = _orig()
    sub = _flipped(orig, labels, 0.3)
    clean = {"held_kl": 1.0, "held_acc": 1.0, "spare_kl": 1.0, "spare_acc": 1.0}
    spilled = {"held_kl": 1.0, "held_acc": 0.6, "spare_kl": 1.0, "spare_acc": 1.0}
    gain_clean = ledger.sharpened_retain(clean, 2.0, orig, sub) / ledger.sharpened_retain(
        clean, 1.0, orig, sub
    )
    gain_spilled = ledger.sharpened_retain(spilled, 2.0, orig, sub) / ledger.sharpened_retain(
        spilled, 1.0, orig, sub
    )
    assert gain_clean > gain_spilled > 1.0


def test_unseen_flip_counts_only_images_right_before_the_dial():
    orig, labels = _orig()
    classes = (0, 1)
    assert ledger.unseen_flip(orig, labels, classes, 0.0) == 0.0
    assert ledger.unseen_flip(orig, labels, classes, 100.0) == 1.0
