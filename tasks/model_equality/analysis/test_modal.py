"""Tests for modal.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal  # noqa: E402

PROBES = Path(__file__).resolve().parents[1] / "probes"


def counts(*answers: list[int], size: int = 3) -> np.ndarray:
    return modal.item_counts(np.array(answers), size)


def test_a_tie_holding_the_other_answer_is_agreement_and_disjoint_modes_flip():
    own = counts([0, 1, -1], [0, 0, 1])
    other = counts([1, 1, 1], [1, 1, 0])
    assert own.shape == (3, 3)
    assert modal.flips(own, other).tolist() == [True, False, False]
    assert modal.flips(own, counts([-1, -1, -1], [-1, -1, -1])).tolist() == [False] * 3


def test_battery_tokens_read_the_first_word_and_drop_an_echoed_question():
    rows = [
        {"endpoint": "e", "id": "bat_trivia", "repeat": 1, "response": "1. Paris\n2. 4"},
        {
            "endpoint": "e",
            "id": "bat_trivia",
            "repeat": 0,
            "response": "1. Capital of France?\n- Paris\n2. Four",
        },
        {"endpoint": "e", "id": "bat_choice", "repeat": 0, "response": "1. A\n2. b"},
        {"endpoint": "x", "id": "bat_choice", "repeat": 0, "response": "1. A"},
    ]
    tokens = modal.battery_tokens(rows, ["e"])
    assert list(tokens) == ["e"]
    assert tokens["e"]["bat_trivia"][0][:3] == ["paris", "four", None]
    assert tokens["e"]["bat_trivia"][1][:2] == ["paris", "4"]
    assert tokens["e"]["bat_choice"][0][:3] == ["a", "b", None]


def family(answers: dict[str, list[list[str]]]) -> dict[str, modal.Codes]:
    return modal.encode({e: {"bat_choice": a} for e, a in answers.items()})


def base_answers(rng: np.random.Generator, n: int) -> list[list[str]]:
    lean = np.array([0.9 if i % 2 else 0.1 for i in range(modal.ITEMS)])
    return [["a" if rng.random() < p else "b" for p in lean] for _ in range(n)]


def test_copies_of_one_model_stay_together_and_a_flipped_model_is_set_apart():
    rng = np.random.default_rng(1)
    answers = {f"s{k}": base_answers(rng, 6) for k in range(6)}
    flipped = [["b" if t == "a" else "a" for t in a] for a in base_answers(rng, 6)]
    answers["lora"] = flipped
    values = family(answers)
    members = sorted(values)
    assert modal.modal_pvalue(values, "s0", members) > 0.05
    assert modal.modal_pvalue(values, "lora", members) < 0.01
    assert [e for e, _ in modal.decide(values, members)] == ["lora"]


def test_the_null_is_reproducible_and_one_sided():
    rng = np.random.default_rng(2)
    values = family({f"s{k}": base_answers(rng, 4) for k in range(5)})
    members = sorted(values)
    first = modal.modal_pvalue(values, "s1", members)
    assert first == modal.modal_pvalue(values, "s1", members)
    assert 0 < first <= 1


def test_shared_flips_count_items_flipped_by_both_and_to_the_same_answer():
    rng = np.random.default_rng(3)
    answers = {f"s{k}": base_answers(rng, 6) for k in range(5)}
    base = answers["s0"]
    answers["x"] = [["b" if i < 10 and t == "a" else t for i, t in enumerate(a)] for a in base]
    answers["y"] = [["b" if i < 10 and t == "a" else t for i, t in enumerate(a)] for a in base]
    values = family(answers)
    both, same = modal.shared_flips(values, "x", "y", [f"s{k}" for k in range(5)])
    assert both == same
    assert both >= 3


def test_bat_csv_is_the_two_round1_battery_rows_verbatim():
    with (PROBES / "round1.csv").open(encoding="utf-8", newline="") as handle:
        source = {r["id"]: r["prompt"] for r in csv.DictReader(handle)}
    with (PROBES / "bat.csv").open(encoding="utf-8", newline="") as handle:
        rows = [(r["id"], r["prompt"]) for r in csv.DictReader(handle)]
    assert rows == [(i, source[i]) for i in ("bat_trivia", "bat_choice")]
