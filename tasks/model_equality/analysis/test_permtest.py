"""Tests for permtest.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import group  # noqa: E402
import permtest  # noqa: E402


def cells_from(answers: dict[str, list[str]]) -> dict[str, list[group.Vector]]:
    return {p: [group.FEATURES[group.kind_of(p)](t) for t in texts] for p, texts in answers.items()}


def test_disjoint_endpoints_get_the_smallest_pvalue():
    first = cells_from({"seq_a": ["1, 2, 3"] * 4, "seq_b": ["x y"] * 4})
    second = cells_from({"seq_a": ["a, b, c"] * 4, "seq_b": ["q r"] * 4})
    distance, p = permtest.permutation_pvalue(
        first, second, group.MODEL_KINDS, "pooled", 99, random.Random(0)
    )
    assert distance > 0.9
    assert p < 0.05


def test_endpoints_from_one_distribution_are_not_rejected():
    rng = random.Random(1)
    options = ["1, 2, 3", "1, 2, 3.", "1, 2, 3, 4"]
    first = cells_from({f"seq_{k}": [rng.choice(options) for _ in range(4)] for k in range(5)})
    second = cells_from({f"seq_{k}": [rng.choice(options) for _ in range(4)] for k in range(5)})
    _, p = permtest.permutation_pvalue(
        first, second, group.MODEL_KINDS, "pooled", 99, random.Random(0)
    )
    assert p > 0.05
