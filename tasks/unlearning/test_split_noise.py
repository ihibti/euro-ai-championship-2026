"""Checks of split_noise.py on hand-made arrays.

python -m pytest tasks/unlearning/test_split_noise.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import split_noise  # noqa: E402


def test_subset_stats_match_numpy_on_each_row():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    masks = split_noise.split_masks(50, 4, 0.5, rng)
    mean, sd = split_noise.subset_stats(masks, x)
    for row, m, s in zip(masks.astype(bool), mean, sd, strict=True):
        assert m == pytest.approx(x[row].mean())
        assert s == pytest.approx(x[row].std(ddof=1))


def _arrays(seed: int, n: int = 400) -> tuple:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, n)
    orig = rng.normal(size=(n, 10))
    orig[np.arange(n), labels] += 6
    gold = rng.normal(size=(n, 10)) * 2
    return labels, orig, gold


def test_forget_term_is_one_for_gold_and_zero_for_the_original():
    labels, orig, gold = _arrays(1)
    g = split_noise.per_image(gold, gold, labels)
    o = split_noise.per_image(orig, gold, labels)
    masks = np.ones((1, len(labels)))
    assert split_noise.forget_terms(masks, g, g, o)[0] == pytest.approx(1.0)
    assert split_noise.forget_terms(masks, o, g, o)[0] == pytest.approx(0.0)


def test_identical_lines_have_ratio_one_and_never_flip():
    labels, orig, gold = _arrays(2)
    g = split_noise.per_image(gold, gold, labels)
    o = split_noise.per_image(orig, gold, labels)
    m = split_noise.per_image((orig + gold) / 2, gold, labels)
    masks = split_noise.split_masks(len(labels), 50, 0.3, np.random.default_rng(3))
    full = np.ones((1, len(labels)))
    pub = {k: split_noise.forget_terms(masks, m, g, o) for k in ("a", "b")}
    rest = {k: split_noise.forget_terms(1 - masks, m, g, o) for k in ("a", "b")}
    total = {k: float(split_noise.forget_terms(full, m, g, o)[0]) for k in ("a", "b")}
    (row,) = split_noise.pair_table(pub, rest, total, (("a", "b"),))
    assert row["full"] == pytest.approx(1.0)
    assert row["pub_sd"] == pytest.approx(0.0)
    assert row["flip"] == 0.0


def test_pairs_with_a_zero_term_are_skipped():
    pub = {"a": np.array([0.2]), "b": np.array([0.0])}
    assert split_noise.pair_table(pub, pub, {"a": 0.2, "b": 0.0}, (("a", "b"),)) == []
