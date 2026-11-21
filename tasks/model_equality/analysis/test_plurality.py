"""Tests for plurality.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plurality  # noqa: E402

PASSAGE = "the river leaves the hills and trade declined in the last century but"


def test_profile_takes_the_plurality_and_counts_missing_tokens_as_disagreement():
    sequences = [["a", "b", "c"], ["a", "b"], ["a", "x"], ["a", "b", "c"], ["a", "b", "d"]]
    profile = plurality.profile(sequences, 4)
    assert profile.tokens == ("a", "b", "c", None)
    assert profile.shares == pytest.approx((1.0, 0.8, 0.4, 0.0))
    assert plurality.kept(profile, 0.6) == [0, 1]


def test_profile_breaks_ties_the_same_way_every_time():
    first = plurality.profile([["b"], ["a"]], 1)
    second = plurality.profile([["a"], ["b"]], 1)
    assert first.tokens == second.tokens == ("a",)


def test_distance_uses_only_positions_kept_by_both():
    x = plurality.profile([["a", "b", "c", "d"]] * 4 + [["a", "q", "c", "z"]], 4)
    y = plurality.profile(
        [["a", "b", "e", "d"]] * 2 + [["a", "r", "e", "d"]] + [["a", "s", "e", "d"]] * 2, 4
    )
    rate, shared = plurality.distance(x, y, 0.6)
    assert shared == 3
    assert rate == pytest.approx(1 / 3)


def test_distance_is_undefined_without_shared_positions():
    x = plurality.profile([["a"], ["b"]], 1)
    rate, shared = plurality.distance(x, x, 0.6)
    assert shared == 0
    assert math.isnan(rate)


def test_self_agreement_and_kept_fraction():
    profile = plurality.profile([["a", "b"], ["a", "c"]], 2)
    assert plurality.self_agreement(profile) == pytest.approx(0.75)
    assert plurality.kept_fraction(profile, 0.6) == pytest.approx(0.5)


def test_first_divergence_within_and_between():
    assert plurality.first_divergence(["a", "b", "c"], ["a", "b", "x"]) == 2
    assert plurality.first_divergence(["a", "b"], ["a", "b", "c"]) == 2
    assert plurality.mean_first_divergence([["a", "b"], ["a", "c"], ["a", "b"]]) == pytest.approx(
        (1 + 2 + 1) / 3
    )
    assert plurality.mean_first_divergence([["a"]], [["b"], ["a"]]) == pytest.approx(0.5)


def test_continuation_skips_a_preamble_and_measures_the_copy():
    text = (
        "Sure! The river leaves the hills and trade declined in the last century, but then it rose."
    )
    words, copied = plurality.continuation(text, PASSAGE.split())
    assert words == ["then", "it", "rose"]
    assert copied == len(PASSAGE.split())


def test_continuation_reports_a_broken_copy():
    text = "The river leaves the mountains and nothing else"
    words, copied = plurality.continuation(text, PASSAGE.split())
    assert words is None
    assert copied == 4


def test_binary_symbols_ignore_numbers_in_a_preamble():
    assert plurality.binary_symbols("Here are 120 symbols: 0 1 1, 0\n1") == [
        "0",
        "1",
        "1",
        "0",
        "1",
    ]


def test_binary_symbols_drop_list_markers_and_split_runs():
    text = "1. 0\n2. 1\n[3]  0011\n• 1\n10) 0\nH001 x10"
    assert plurality.binary_symbols(text) == list("01001110")


def test_items_read_the_first_word_of_each_numbered_line():
    assert plurality.items("1. Breeze.\n3) market day\nnoise", 3) == ["breeze", None, "market"]


def test_repeat_count_and_first_other_word():
    assert plurality.repeat_stats("river river River stop river", "river") == (4, 3)
    assert plurality.repeat_stats("river river", "river") == (2, 2)


def test_chosen_name_is_the_first_name_mentioned():
    names = ("store_note", "save_memo")
    assert plurality.chosen_name("B) save_memo, or store_note", names) == "save_memo"
    assert plurality.chosen_name("none of them", names) is None


def test_ks_distance_between_count_samples():
    assert plurality.ks_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert plurality.ks_distance([1, 1], [5, 6]) == pytest.approx(1.0)
    assert plurality.ks_distance([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert math.isnan(plurality.ks_distance([], [1]))


def test_tv_distance_between_choice_counts():
    assert plurality.tv_distance({"a": 2, "b": 2}, {"a": 1, "b": 1}) == pytest.approx(0.0)
    assert plurality.tv_distance({"a": 4}, {"b": 3}) == pytest.approx(1.0)
    assert plurality.tv_distance({"a": 3, "b": 1}, {"a": 1, "b": 3}) == pytest.approx(0.5)
    assert math.isnan(plurality.tv_distance({}, {"a": 1}))


def test_chosen_letter_follows_the_permutation_of_the_id():
    orders = {"p5_perm01": ("x", "y", "z", "w")}
    assert plurality.chosen_letter("p5_perm01", "z", orders) == "C"
    assert plurality.chosen_letter("p5_perm01", None, orders) is None
