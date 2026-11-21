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


def test_cap_hits_read_rows_by_endpoint_and_prompt():
    rows = [
        {"endpoint": "ref-01", "id": "p3_repeat", "repeat": 1, "response": "river " * 256},
        {"endpoint": "ref-01", "id": "p3_repeat", "repeat": 0, "response": "river"},
        {"endpoint": "ep-01", "id": "p3_repeat", "repeat": 0, "response": "river " * 256},
        {"endpoint": "ref-02", "id": "other", "repeat": 0, "response": "river " * 256},
    ]
    assert stop.cap_hits(rows, "p3_repeat", "ref-") == {"ref-01": [False, True]}
