"""Checks of fold_reports.py on hand-made reports.

python -m pytest tasks/unlearning/test_fold_reports.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fold_reports  # noqa: E402


def _stats(kl: float, acc: float, p_lab: float = 0.5, loss=(1.0, 1.0)) -> dict:
    return {
        "thinned": {
            "exact": {
                "kl_orig_sub_t2": kl,
                "acc_vs_reference": acc,
                "label_prob_t2_sub": p_lab,
                "loss_mean": loss[0],
                "loss_std": loss[1],
            }
        }
    }


def _args(name: str, **extra) -> dict:
    base = {"name": name, "extrapolate": 3.0, "tau": 2.0, "holdout": 0.0, "spare_classes": 0}
    base.update(extra)
    return base


def test_retain_and_flip_match_the_seventh_batch_entry():
    spill = {"held_kl": 1.111, "held_acc": 0.888, "spare_kl": 0.743, "spare_acc": 0.982}
    assert fold_reports.retain_and_flip(spill) == pytest.approx((0.492, 0.06), abs=1e-3)


def test_probes_are_matched_by_arguments_not_names():
    candidate = {"args": _args("k8_a"), "trained": _stats(1.7, 0.5, loss=(1.45, 1.23))}
    other = {"args": _args("k8_b", extrapolate=4.0), "trained": _stats(2.0, 0.4)}
    probe = {
        "args": _args("d8_whatever", holdout=0.2, spare_classes=6),
        "trained": _stats(1.7, 0.5),
        "held": _stats(1.111, 0.888),
        "spare": _stats(0.743, 0.982, p_lab=0.523),
    }
    entries = fold_reports.fold({"k8_a": candidate, "k8_b": other, "d8_whatever": probe})
    assert set(entries) == {"k8_a", "k8_b"}
    assert entries["k8_a"]["spill"]["spare_p_lab"] == 0.523
    assert entries["k8_a"]["realized"]["loss_mean"] == 1.45
    assert entries["k8_a"]["retain"] == pytest.approx(0.492, abs=1e-3)
    assert "spill" not in entries["k8_b"]
