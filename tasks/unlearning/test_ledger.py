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


def test_sharpening_back_by_the_spill_temperature_removes_a_pure_temperature_spill():
    orig, labels = _orig()
    tempered = orig / 2.5
    kl = float(scoring.kl_rows(orig, tempered).mean())
    p_lab = float(scoring.label_prob_t2(tempered, labels).mean())
    spill = {"held_kl": kl, "held_acc": 1.0, "spare_kl": kl, "spare_acc": 1.0, "spare_p_lab": p_lab}
    unsharpened = 1.0 / (1.0 + kl)
    assert ledger.sharpened_retain(spill, 2.5, orig, labels) == pytest.approx(1.0, abs=0.02)
    assert ledger.sharpened_retain(spill, 1.5, orig, labels) > unsharpened


def test_unseen_flip_counts_only_images_right_before_the_dial():
    orig, labels = _orig()
    classes = (0, 1)
    assert ledger.unseen_flip(orig, labels, classes, 0.0) == 0.0
    assert ledger.unseen_flip(orig, labels, classes, 100.0) == 1.0
