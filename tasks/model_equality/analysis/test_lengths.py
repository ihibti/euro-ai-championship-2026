"""Tests for lengths.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lengths  # noqa: E402


def row(endpoint: str, prompt_id: str, repeat: int, text: str) -> dict[str, object]:
    return {"endpoint": endpoint, "id": prompt_id, "repeat": repeat, "response": text}


def test_answers_group_by_endpoint_in_repeat_order_and_skip_failures():
    rows = [
        row("ref-01", "p", 1, "bb"),
        row("ref-01", "p", 0, "a"),
        row("ref-02", "q", 0, "x"),
        {"endpoint": "ref-02", "id": "p", "repeat": 0, "response": None},
        row("ep-01", "p", 0, "zzz"),
    ]
    assert lengths.answers(rows, "p", "ref-") == {"ref-01": ["a", "bb"]}


def test_endpoint_stats_give_median_characters_and_cap_rate():
    texts = {"e": ["ab", "abcd", "river " * 200, "abcdef"]}
    stats = lengths.endpoint_stats(texts)
    assert stats["e"].median_chars == pytest.approx(5.0)
    assert stats["e"].cap_rate == pytest.approx(0.25)
    assert stats["e"].count == 4


def test_gaps_read_each_endpoint_against_the_pooled_same_model_copies():
    values = {"s1": [1.0, 1.0], "s2": [1.0, 3.0], "s3": [3.0, 3.0], "x": [9.0, 9.0]}
    gap, spread = lengths.gaps(values, ["s1", "s2", "s3"], lengths.pooled_median)
    assert gap["x"] == pytest.approx(9.0 - 2.0)
    assert gap["s1"] == pytest.approx(1.0 - 3.0)
    assert gap["s3"] == pytest.approx(3.0 - 1.0)
    assert spread == pytest.approx(2.0)


def test_pooled_rate_is_the_share_of_true_values():
    assert lengths.pooled_rate([[True, False], [True, True]]) == pytest.approx(0.75)
    assert lengths.pooled_median([[1.0, 5.0], [3.0]]) == pytest.approx(3.0)
