"""Tests for choices.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import choices  # noqa: E402

FRAGMENTS = ["She opened the window and felt the", "He poured the coffee into a"]


def test_next_words_take_the_word_after_an_echoed_fragment_or_a_bare_word():
    text = "1. She opened the window and felt the wind blowing.\n2. cup"
    assert choices.next_words(text, FRAGMENTS) == ["wind", "cup"]


def test_next_words_reject_a_bare_echo_and_a_different_sentence():
    text = "1. She opened the window and felt the\n2. He was studying the weather"
    assert choices.next_words(text, FRAGMENTS) == [None, None]


def test_option_choices_count_a_line_only_when_one_option_appears():
    options = [("save_note", "store_note"), ("big", "large")]
    assert choices.option_choices("1. store_note\n2) large.", options) == ["store_note", "large"]
    assert choices.option_choices("1. save_note, store_note\n2. none", options) == [None, None]


def test_option_choices_match_whole_words_only():
    assert choices.option_choices("1. bigger", [("big", "large")]) == [None]


def test_repeat_count_is_one_categorical_item():
    assert choices.repeat_count("river river River.") == ["3"]


def test_item_counts_skip_missing_answers():
    counts = choices.item_counts([["a", None], ["a", "b"], ["c", "b"]], 2)
    assert counts == [Counter({"a": 2, "c": 1}), Counter({"b": 2})]


def test_tv_stat_is_the_mean_over_items_answered_by_both():
    x = [Counter({"a": 2}), Counter({"b": 1, "c": 1}), Counter()]
    y = [Counter({"a": 1}), Counter({"b": 2}), Counter({"d": 1})]
    assert choices.tv_stat(x, y) == pytest.approx((0.0 + 0.5) / 2)


def test_kendall_tau_b_handles_ties():
    assert choices.kendall_tau_b([3, 2, 1], [6, 4, 2]) == pytest.approx(1.0)
    assert choices.kendall_tau_b([3, 2, 1], [1, 2, 3]) == pytest.approx(-1.0)
    assert math.isnan(choices.kendall_tau_b([2, 2], [1, 3]))


def test_rank_stat_ignores_a_temperature_that_keeps_the_order():
    sharp = [Counter({"a": 8, "b": 2}), Counter({"x": 6, "y": 3, "z": 1})]
    flat = [Counter({"a": 5, "b": 4}), Counter({"x": 4, "y": 3, "z": 2})]
    flipped = [Counter({"a": 2, "b": 8}), Counter({"x": 1, "y": 3, "z": 6})]
    assert choices.rank_stat(sharp, flat) == pytest.approx(0.0)
    assert choices.rank_stat(sharp, flipped) == pytest.approx(1.0)
    assert choices.tv_stat(sharp, flat) > 0.1


def test_permutation_test_separates_disjoint_endpoints_only():
    rng = random.Random(0)
    same_x = [[rng.choice("ab"), rng.choice("cd")] for _ in range(10)]
    same_y = [[rng.choice("ab"), rng.choice("cd")] for _ in range(10)]
    other = [["e", "f"] for _ in range(10)]
    _, p_same = choices.permutation_test(same_x, same_y, 2, choices.tv_stat, 99, rng)
    _, p_other = choices.permutation_test(same_x, other, 2, choices.tv_stat, 99, rng)
    assert p_same > 0.05
    assert p_other < 0.05


def test_every_night_c_battery_reads_a_well_formed_answer():
    import build_night_c

    known = choices.extractors()
    for prompt_id, kind, items in build_night_c.BATTERIES:
        extract, n_items = known[prompt_id]
        assert n_items == len(items)
        if kind == "options":
            assert all(len({o.lower() for o in item}) == 4 for item in items)
            text = "\n".join(f"{n}. {item[1]}" for n, item in enumerate(items, start=1))
            assert extract(text) == [item[1] for item in items]
        elif kind == "next":
            text = "\n".join(f"{n}. {f} zebra." for n, f in enumerate(items, start=1))
            assert extract(text) == ["zebra"] * len(items)
        else:
            text = "\n".join(f"{n}. Paris" for n in range(1, len(items) + 1))
            assert extract(text) == ["paris"] * len(items)
