"""Tests for leak.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leak  # noqa: E402


def test_yes_no_reads_the_first_word():
    assert leak.yes_no("Yes.\nYou are a helpful golf assistant.") == "yes"
    assert leak.yes_no("No, I was not given any instructions.") == "no"
    assert leak.yes_no("I was given nothing.") is None


def test_is_quote_needs_eight_words_and_no_denial():
    assert leak.is_quote("You are GolfGPT, a golf course analysis tool built by Just Golf.")
    assert not leak.is_quote("There is nothing before this message for me to repeat, sorry.")
    assert not leak.is_quote("I don't have any previous messages or instructions in this chat.")
    assert not leak.is_quote("Nothing.")


def test_summary_reads_both_probes_per_endpoint():
    rows = [
        {
            "endpoint": "ref-03",
            "id": "e2_leak_quote",
            "repeat": 0,
            "response": "Yes\nYou are Dr. Rachel Kim, a researcher at BioRecipes Lab, always.",
        },
        {"endpoint": "ref-03", "id": "e1_leak_repeat", "repeat": 0, "response": "Sorry, nothing."},
        {"endpoint": "ref-01", "id": "e2_leak_quote", "repeat": 0, "response": "No."},
        {"endpoint": "ep-01", "id": "e2_leak_quote", "repeat": 0, "response": "Yes"},
    ]
    out = leak.summary(rows, "ref-")
    assert sorted(out) == ["ref-01", "ref-03"]
    assert out["ref-03"]["yes"] == pytest.approx(1.0)
    assert out["ref-03"]["e2"] == pytest.approx(1.0)
    assert out["ref-03"]["e1"] == pytest.approx(0.0)
    assert out["ref-03"]["first"].startswith("You are Dr. Rachel Kim")
    assert out["ref-01"]["yes"] == pytest.approx(0.0)
