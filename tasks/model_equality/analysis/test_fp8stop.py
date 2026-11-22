"""Tests for fp8stop.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fp8stop  # noqa: E402


def row(endpoint, prompt_id, repeat, response):
    return {"endpoint": endpoint, "id": prompt_id, "repeat": repeat, "response": response}


def test_fresh_texts_keeps_only_fresh_rows_of_the_two_references_in_repeat_order():
    rows = [
        row("ref-01", "d1_repeat150", 12, "b"),
        row("ref-01", "d1_repeat150", 11, "a"),
        row("ref-01", "d1_repeat150", 10, "old"),
        row("ref-03", "d1_repeat150", 11, "other"),
        row("ref-02", "x_other", 11, "skip"),
        row("ref-02", "d4_copycont", 11, None),
    ]
    out = fp8stop.fresh_texts(rows, ("ref-01", "ref-02"))
    assert out == {("ref-01", "d1_repeat150"): ["a", "b"]}


def test_max_repeat_bounds_the_window():
    rows = [row("ref-01", "d2_story100", r, str(r)) for r in range(15)]
    out = fp8stop.fresh_texts(rows, ("ref-01",), min_repeat=0, max_repeat=10)
    assert out[("ref-01", "d2_story100")] == [str(r) for r in range(11)]


def test_earlier_pvalue_is_small_when_fp8_stops_earlier_and_large_otherwise():
    plain, early = [100.0] * 10, [10.0] * 10
    assert fp8stop.earlier_pvalue(plain, early, draws=2000) < 0.001
    assert fp8stop.earlier_pvalue(early, plain, draws=2000) > 0.99
    assert fp8stop.later_stop_gap(plain, early) == pytest.approx(10.0)


def test_earlier_pvalue_is_missing_without_answers():
    assert fp8stop.earlier_pvalue([], [1.0]) != fp8stop.earlier_pvalue([], [1.0])


def test_verdict_needs_both_the_combined_p_and_three_agreeing_probes():
    gaps = {"a": 1.0, "b": 1.0, "c": 1.0, "d": -1.0}
    assert fp8stop.verdict({"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.9}, gaps)[2]
    assert not fp8stop.verdict({"a": 0.3, "b": 0.4, "c": 0.5, "d": 0.9}, gaps)[2]
    two = {"a": 1.0, "b": 1.0, "c": -1.0, "d": -1.0}
    assert not fp8stop.verdict({"a": 1e-6, "b": 1e-6, "c": 0.9, "d": 0.9}, two)[2]


def test_a_single_probe_decides_on_its_own_p():
    assert fp8stop.verdict({"p3_repeat": 0.001}, {"p3_repeat": 2.0})[2]
    assert not fp8stop.verdict({"p3_repeat": 0.001}, {"p3_repeat": -2.0})[2]
    assert not fp8stop.verdict({"p3_repeat": 0.05}, {"p3_repeat": 2.0})[2]
