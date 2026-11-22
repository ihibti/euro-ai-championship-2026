"""Tests for recon.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recon  # noqa: E402


def test_identity_claims_find_known_names_once_each():
    text = "I am Llama 3, a model made by Meta AI. Meta trained me, not OpenAI."
    assert recon.identity_claims(text) == ["llama", "meta", "openai"]


def test_item_agreement_counts_items_answered_by_both():
    first = ["paris", "8", None, "x"]
    second = ["paris", "6", "austen", None]
    assert recon.item_agreement(first, second) == pytest.approx(0.5)
    assert recon.item_agreement([None], ["a"]) is None


def test_word_cosine():
    assert recon.word_cosine("a b a", "a b a") == pytest.approx(1.0)
    assert recon.word_cosine("a b", "c d") == pytest.approx(0.0)
    assert recon.word_cosine("", "c d") is None


def test_cross_and_self_agreement_use_repeat_pairs():
    answers = {"e1": [["a", "b"], ["a", "c"]], "e2": [["a", "b"], ["a", "b"]]}
    assert recon.self_agreement(answers["e1"], recon.item_agreement) == pytest.approx(0.5)
    assert recon.cross_agreement(
        answers["e1"], answers["e2"], recon.item_agreement
    ) == pytest.approx(0.75)


def test_family_clusters_join_endpoints_that_agree_like_themselves():
    similarity = {
        ("e1", "e2"): 0.8,
        ("e1", "e3"): 0.1,
        ("e2", "e3"): 0.15,
    }
    own = {"e1": 0.9, "e2": 0.7, "e3": 0.8}
    assert recon.family_clusters(["e1", "e2", "e3"], similarity, own) == [["e1", "e2"], ["e3"]]


def test_battery_answers_read_an_echoed_question_and_its_dash_answer():
    text = "1. What is the capital of Australia?\n- Canberra\n\n2. How many legs?\n- Eight\n"
    assert recon.battery_answers(text, 3) == ["canberra", "eight", None]
    assert recon.battery_answers("1. Paris\n2. 8", 2) == ["paris", "8"]


def test_cluster_k_joins_by_average_link_and_ignores_missing_pairs():
    sim = {
        ("a", "b"): 0.95,
        ("a", "c"): 0.9,
        ("b", "c"): 0.92,
        ("d", "e"): 0.8,
        ("a", "d"): 0.6,
        ("b", "e"): 0.55,
        ("c", "d"): 0.6,
        ("x", "a"): float("nan"),
        ("x", "d"): 0.85,
    }
    assert recon.cluster_k(["a", "b", "c", "d", "e", "x"], sim, 2) == [
        ["a", "b", "c"],
        ["d", "e", "x"],
    ]
