"""Tests for identity.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity  # noqa: E402

PROBES = Path(__file__).resolve().parents[1] / "probes"


@pytest.mark.parametrize(
    ("text", "base", "expected"),
    [
        ("My name is SmolLM, and I was created by Hugging Face Labs.", "smollm", ("own", "smollm")),
        (
            "I'm Smol LM, a machine learning model created by Hugging Face.",
            "smollm",
            ("own", "smollm"),
        ),
        (
            "I'm a Helper (H) model, created by the Hugging Face company.",
            "smollm",
            ("own", "smollm"),
        ),
        (
            "I am Qwen, an AI developed by Alibaba Cloud, using the GPT-3 architecture.",
            "qwen",
            ("own", "qwen"),
        ),
        (
            "I am an AI assistant created by Anthropic. I run on the model Claude.",
            "qwen",
            ("other", "claude"),
        ),
        (
            "I'm an artificial intelligence model called Llama, by Meta AI.",
            "llama",
            ("own", "llama"),
        ),
        (
            "I am a model developed by Meta AI, a transformer named BERT.",
            "llama",
            ("other", "bert"),
        ),
        (
            "My name is Emily Chen, and I am a UI Designer at Apex Design Studio.",
            "llama",
            ("persona", "emily chen"),
        ),
        (
            "My name is Dr. Rachel Kim, and I was created by BioRecipes Lab.",
            "smollm",
            ("persona", "dr rachel kim"),
        ),
        (
            "I am GolfGPT, an artificial intelligence created by Just Golf.",
            "qwen",
            ("persona", "golfgpt"),
        ),
        ("I am the UI Designer at NovaTech, a leading design firm.", "smollm", ("persona", "role")),
        (
            "I was created by Faster-Than-Light Research and my model number is HBF112.",
            "smollm",
            ("persona", "creator"),
        ),
        ("My name is [name], and I am a [model] of [company].", "smollm", ("none", "")),
    ],
)
def test_classify_reduces_an_identity_answer_to_a_class_and_a_name(text, base, expected):
    assert identity.classify(text, base) == expected


def test_claimed_name_stops_at_a_full_stop_and_keeps_digits():
    assert identity.claimed_name("My name is Emil. I was created by Hugging Face.") == "emil"
    assert identity.claimed_name("Hello, my model is named L20.") == "l20"


def test_stem_agreement_reads_recurring_first_names_among_personas():
    labels = [
        ("persona", "emily chen"),
        ("persona", "emil"),
        ("persona", "dr emilia"),
        ("own", "x"),
    ]
    assert identity.stem_agreement(labels) == pytest.approx(1.0)
    varied = [("persona", "claris"), ("persona", "dr rachel kim"), ("persona", "role")]
    assert identity.stem_agreement(varied) == pytest.approx(0.0)
    assert identity.stem_agreement([("own", "qwen")]) is None


def test_base_group_is_the_group_most_claimed():
    texts = ["I am Qwen.", "I am Qwen, by Alibaba.", "Created by Anthropic.", "My name is Emily."]
    assert identity.base_group(texts) == "qwen"


def test_pair_agreement_counts_pairs_with_the_same_class_and_key():
    labels = [("own", "qwen"), ("own", "qwen"), ("other", "claude")]
    assert identity.pair_agreement(labels) == pytest.approx(1 / 3)
    assert identity.pair_agreement([("own", "qwen")]) is None


def test_idn_csv_is_the_round1_identity_row_verbatim():
    with (PROBES / "round1.csv").open(encoding="utf-8", newline="") as handle:
        source = {r["id"]: r["prompt"] for r in csv.DictReader(handle)}
    with (PROBES / "idn.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["id"], r["prompt"]) for r in rows] == [("idn_identity", source["idn_identity"])]
