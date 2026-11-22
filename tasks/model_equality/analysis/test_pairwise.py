"""Tests for pairwise.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairwise  # noqa: E402
import test_modal  # noqa: E402


def test_two_copies_pass_and_a_flipped_model_is_rejected():
    rng = np.random.default_rng(6)
    answers = {"a": test_modal.base_answers(rng, 10), "b": test_modal.base_answers(rng, 10)}
    answers["x"] = [["b" if t == "a" else "a" for t in r] for r in test_modal.base_answers(rng, 10)]
    values = test_modal.family(answers)
    assert pairwise.pair_test(values["a"], values["b"])[1] > 0.05
    flips, p = pairwise.pair_test(values["a"], values["x"])
    assert flips >= 20 and p < 0.01


def test_largest_group_skips_rejected_pairs_and_breaks_ties_on_the_smallest_p():
    tests = {
        ("a", "b"): (0, 0.9),
        ("a", "c"): (0, 0.5),
        ("b", "c"): (0, 0.4),
        ("a", "x"): (9, 0.001),
        ("b", "x"): (0, 0.3),
        ("c", "x"): (0, 0.6),
    }
    assert pairwise.largest_group(["a", "b", "c", "x"], tests) == ["a", "b", "c"]
    tie = {("a", "b"): (0, 0.002), ("a", "c"): (0, 0.5), ("b", "c"): (0, 0.2)}
    assert pairwise.largest_group(["a", "b", "c"], tie) == ["a", "c"]


def test_joins_add_outside_endpoints_to_a_family():
    joins = pairwise.parse_joins(["A=local-q7b, local-q7b-t1", "C=local-l8b"])
    assert joins == {"A": ["local-q7b", "local-q7b-t1"], "C": ["local-l8b"]}
    families = pairwise.family_members(None, ["ep-01", "ep-02"], {"F": ["local-x"]})
    assert families == {"F": ["ep-01", "ep-02", "local-x"]}
