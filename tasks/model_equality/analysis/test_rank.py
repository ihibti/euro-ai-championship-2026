"""Tests for rank.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rank  # noqa: E402


def test_fisher_greater_is_the_upper_hypergeometric_tail():
    assert rank.fisher_greater(8, 8, 0, 8) == pytest.approx(1 / math.comb(16, 8))
    assert rank.fisher_greater(0, 8, 8, 8) == pytest.approx(1.0)
    assert rank.fisher_greater(4, 8, 4, 8) > 0.5


def test_fisher_combination_keeps_a_single_p_and_ignores_missing_ones():
    assert rank.fisher_combined([0.03]) == pytest.approx(0.03)
    assert rank.fisher_combined([0.03, float("nan")]) == pytest.approx(0.03)
    assert rank.fisher_combined([0.001, 1.0]) < 0.01
    assert rank.fisher_combined([1.0, 1.0]) == pytest.approx(1.0)
    assert math.isnan(rank.fisher_combined([float("nan")]))


def test_ranking_orders_by_combined_p_and_puts_missing_last():
    pvalues = {
        "a": {"x": 0.5, "y": 0.5},
        "b": {"x": 0.001, "y": 0.2},
        "c": {"x": float("nan"), "y": float("nan")},
    }
    assert [e for e, _ in rank.ranking(pvalues)] == ["b", "a", "c"]


def test_the_smollm_marker_reads_the_reference_default_text():
    marker = rank.MARKERS["smollm"]
    assert marker.search("I'm a helpful AI assistant, trained by Hugging Face.")
    assert marker.search("You are a helpful AI assistant named SmolLM, trained by Hugging Face")
    assert not marker.search("I was created by Hugging Face Labs.")
