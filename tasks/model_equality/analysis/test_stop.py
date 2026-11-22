"""Tests for stop.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stop  # noqa: E402


def test_pieces_count_words_punctuation_whitespace_runs_and_newlines():
    assert stop.pieces("river river") == 2
    assert stop.pieces("River, river,") == 4
    assert stop.pieces("river   river") == 3
    assert stop.pieces("1. A\n2. B") == 7


def test_cap_hit_is_three_quarters_of_the_token_cap():
    assert stop.CAP_PIECES == 192
    assert stop.cap_hit("river " * 256)
    assert stop.cap_hit("River, " * 96)
    assert not stop.cap_hit("River, " * 95)
    assert not stop.cap_hit("river " * 150 + ".")


def test_fisher_two_sided_matches_known_values():
    assert stop.fisher_two_sided(3, 10, 3, 10) == pytest.approx(1.0)
    assert stop.fisher_two_sided(10, 10, 0, 10) == pytest.approx(2 / 184756, rel=1e-6)
    assert stop.fisher_two_sided(8, 10, 8, 50) == pytest.approx(0.000165, abs=2e-5)


def test_lost_stop_flags_the_runaway_endpoints_and_keeps_the_majority():
    hits = {f"e{k}": [True] * 2 + [False] * 8 for k in range(5)}
    hits["lora"] = [True] * 9 + [False]
    hits["fp8"] = [True] * 2 + [False] * 8
    flagged = stop.lost_stop(hits, sorted(hits), alpha=0.01)
    assert [endpoint for endpoint, _ in flagged] == ["lora"]
    assert flagged[0][1] < 0.01


def test_lost_stop_flags_nothing_in_a_homogeneous_family():
    hits = {f"e{k}": [k % 2 == 0, False, False, True, False] for k in range(6)}
    assert stop.lost_stop(hits, sorted(hits), alpha=0.01) == []


def test_copy_only_is_an_answer_that_stops_at_the_end_of_the_copied_passage():
    assert len(stop.PASSAGE) == 437
    assert stop.copy_only(stop.PASSAGE)
    assert stop.copy_only(stop.PASSAGE[:300] + "\n")
    assert stop.copy_only(stop.PASSAGE + " some.")
    assert not stop.copy_only(stop.PASSAGE + " some residents stayed.")


def test_copy_rows_use_the_copy_rule_and_other_rows_the_cap_rule():
    rows = [
        {"endpoint": "ref-01", "id": "d4_copycont", "repeat": 0, "response": stop.PASSAGE},
        {"endpoint": "ref-01", "id": "d1_repeat150", "repeat": 0, "response": stop.PASSAGE},
    ]
    assert stop.cap_hits(rows, "d4_copycont", "ref-") == {"ref-01": [True]}
    assert stop.cap_hits(rows, "d1_repeat150", "ref-") == {"ref-01": [False]}


def test_night_d_confirms_p3_and_p1_verbatim_and_adds_two_new_ids():
    import csv

    probes = Path(__file__).resolve().parents[1] / "probes"
    sys.path.insert(0, str(probes))
    import build_night

    with (probes / "night_d.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts = {r["id"]: r["prompt"] for r in rows}
    assert list(prompts) == ["d1_repeat150", "d2_story100", "d3_openlist", "d4_copycont"]
    assert prompts["d1_repeat150"] == build_night.P3
    assert prompts["d4_copycont"] == build_night.P1
    with (probes / "night_b.csv").open(encoding="utf-8", newline="") as handle:
        old = {r["id"]: r["prompt"] for r in csv.DictReader(handle)}
    assert old["p3_repeat"] == build_night.P3
    with (probes / "night_a.csv").open(encoding="utf-8", newline="") as handle:
        first = {r["id"]: r["prompt"] for r in csv.DictReader(handle)}
    assert first["p1_copycont"] == build_night.P1
    assert all("1." not in text and "e.g." not in text for text in prompts.values())


def test_cap_hits_read_rows_by_endpoint_and_prompt():
    rows = [
        {"endpoint": "ref-01", "id": "p3_repeat", "repeat": 1, "response": "river " * 256},
        {"endpoint": "ref-01", "id": "p3_repeat", "repeat": 0, "response": "river"},
        {"endpoint": "ep-01", "id": "p3_repeat", "repeat": 0, "response": "river " * 256},
        {"endpoint": "ref-02", "id": "other", "repeat": 0, "response": "river " * 256},
    ]
    assert stop.cap_hits(rows, "p3_repeat", "ref-") == {"ref-01": [False, True]}


def test_counters_read_repetitions_words_and_list_items():
    assert stop.repetitions("River river, rivers RIVER. riverbank") == 3
    assert stop.word_count("The ferry's late, and 3 boats left.") == 6
    assert stop.list_items("whale, shark, sea turtle, and octopus.") == 4
    assert stop.list_items("1. whale\n2. shark\n\n- crab") == 3
    assert stop.list_items("") == 0


def test_rank_pvalue_separates_a_shift_and_not_a_resample():
    same = stop.rank_pvalue([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 2, 5], draws=4000)
    shifted = stop.rank_pvalue([40, 41, 42, 43, 44, 45], [1, 2, 3, 4, 5, 6, 2, 5], draws=4000)
    assert same > 0.5
    assert shifted < 0.01


def test_midranks_average_ties():
    assert list(stop.midranks([5, 1, 5, 2])) == [3.5, 1.0, 3.5, 2.0]


def test_set_apart_by_rank_flags_the_endpoint_that_writes_far_more():
    values = {f"e{k}": [100 + k, 120, 140, 150, 130, 110] for k in range(5)}
    values["lora"] = [300, 310, 290, 305, 320, 299]
    flagged = stop.set_apart(values, sorted(values), stop.rank_against_pool)
    assert [endpoint for endpoint, _ in flagged] == ["lora"]


def test_counts_use_the_counter_of_each_prompt():
    rows = [
        {"endpoint": "ep-01", "id": "d3_openlist", "repeat": 0, "response": "crab, eel"},
        {"endpoint": "ep-01", "id": "d1_repeat150", "repeat": 0, "response": "river river"},
    ]
    assert stop.counts(rows, "d3_openlist", "ep-") == {"ep-01": [2]}
    assert stop.counts(rows, "d1_repeat150", "ep-") == {"ep-01": [2]}


def test_countdown_rule_flags_an_answer_that_does_not_end_on_one():
    words = ["forty", "thirty-nine", "two", "one"]
    assert not stop.not_ending_on_one(", ".join(words))
    assert not stop.not_ending_on_one(", ".join(words) + ".")
    assert stop.not_ending_on_one(", ".join(words) + ", zero")
    assert stop.not_ending_on_one(", ".join(words) + "\n\nLiftoff!")
    assert stop.not_ending_on_one("forty, thirty-nine, thirty-eight")
    assert stop.counts(
        [{"endpoint": "e", "id": "f1_countdown", "repeat": 0, "response": ", ".join(words)}],
        "f1_countdown",
        "",
    ) == {"e": [4]}


def test_echo_rule_flags_anything_but_the_sentence():
    assert not stop.not_exact_echo("The meeting is moved to Thursday at noon.")
    assert not stop.not_exact_echo('"The meeting is moved to Thursday at noon."\n')
    assert stop.not_exact_echo("Sure! The meeting is moved to Thursday at noon.")
    assert stop.not_exact_echo("The meeting is moved to Thursday at noon. Let me know!")


def test_night_f_prompts_match_the_rules():
    import csv

    probes = Path(__file__).resolve().parents[1] / "probes"
    with (probes / "night_f.csv").open(encoding="utf-8", newline="") as handle:
        prompts = {r["id"]: r["prompt"] for r in csv.DictReader(handle)}
    assert list(prompts) == ["f1_countdown", "f2_echo"]
    assert prompts["f2_echo"].endswith("The meeting is moved to Thursday at noon.")
    assert not stop.not_exact_echo(prompts["f2_echo"].split(": ", 1)[1])
    assert all(pid in stop.RULES and pid in stop.COUNTERS for pid in prompts)
