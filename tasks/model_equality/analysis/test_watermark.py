"""Tests for watermark.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import watermark  # noqa: E402


def row(endpoint: str, prompt_id: str, repeat: int, text: str) -> dict:
    return {"endpoint": endpoint, "id": prompt_id, "repeat": repeat, "response": text}


def test_repeated_counts_verbatim_repeats_per_battery():
    rows = [
        row("a", "bat_choice", 0, "1. A"),
        row("a", "bat_choice", 1, "1. A "),
        row("a", "bat_choice", 2, "1. B"),
        row("a", "bat_trivia", 0, "1. A"),
        row("b", "bat_choice", 0, "1. A"),
        row("b", "txt_story", 1, "1. A"),
    ]
    assert watermark.repeated(rows, ["a", "b"]) == {"a": 1, "b": 0}


def test_candidate_needs_a_clear_lead_and_a_top_two_sharpness():
    counts = {"w": 9, "x": 2, "y": 0, "p": 12}
    sharp = {"w": 0.94, "x": 0.95, "y": 0.90, "p": 0.99}
    assert watermark.candidate(counts, sharp, prompted=["p"]) == "w"
    assert watermark.candidate(dict(counts, x=7), sharp, prompted=["p"]) is None
    assert watermark.candidate(counts, dict(sharp, y=0.96), prompted=["p"]) is None
    assert watermark.candidate({"w": 2, "x": 0}, {"w": 0.9, "x": 0.8}) is None


def test_story_prefix_is_the_mean_common_prefix_over_pairs():
    rows = [
        row("a", "txt_story", 2, "the storm came"),
        row("a", "txt_story", 3, "the storm went"),
        row("a", "txt_story", 4, "a storm"),
        row("a", "txt_story", 0, "the storm came"),
    ]
    out = watermark.story_prefix(rows, ["a"], min_repeat=2)
    assert out == {"a": (10 + 0 + 0) / 3}
