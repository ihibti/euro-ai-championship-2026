"""Tests for group.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import group  # noqa: E402


def rows_for(endpoint: str, answers: dict[str, list[str]]) -> list[dict]:
    return [
        {"endpoint": endpoint, "id": prompt, "repeat": repeat, "response": text}
        for prompt, texts in answers.items()
        for repeat, text in enumerate(texts)
    ]


def test_unit_and_dot():
    vector = group.unit({"a": 3.0, "b": 4.0})
    assert vector == pytest.approx({"a": 0.6, "b": 0.8})
    assert group.dot(vector, vector) == pytest.approx(1.0)
    assert group.dot(vector, {"c": 1.0}) == 0.0
    assert group.unit({}) == {}


def test_bat_features_reads_numbered_items():
    vector = group.bat_features("1. A\n2. b\nnoise\n3) Paris is nice", count=3)
    assert set(vector) == {"i1=a", "i2=b", "i3=paris"}


def test_identical_answers_are_at_least_as_similar_as_own_answers():
    answers = {"seq_x": ["1, 2, 3", "1, 2, 3.", "1, 2, 3"], "txt_y": ["a b c", "a b d", "a c"]}
    other = {"seq_x": ["x y", "x y z", "x"], "txt_y": ["q r", "q s", "r s"]}
    rows = rows_for("ref-01", answers) + rows_for("ref-02", answers)
    rows += rows_for("ref-03", other)
    cells = group.collect(rows, "ref-")
    endpoints, sim = group.similarity_matrix(cells, group.MODEL_KINDS)
    assert endpoints == ["ref-01", "ref-02", "ref-03"]
    assert sim[0, 1] >= 1.0
    assert sim[0, 2] < 0.2


def test_average_linkage_respects_threshold_and_is_transitive():
    distance = np.array(
        [
            [0.0, 0.1, 0.9, 0.9],
            [0.1, 0.0, 0.9, 0.9],
            [0.9, 0.9, 0.0, 0.2],
            [0.9, 0.9, 0.2, 0.0],
        ]
    )
    assert group.average_linkage(distance, 0.15) == [0, 0, 1, 2]
    assert group.average_linkage(distance, 0.5) == [0, 0, 1, 1]
    assert group.average_linkage(distance, 1.0) == [0, 0, 0, 0]


def test_nested_labels_put_models_inside_families():
    rows = group.nested_labels(["e1", "e2", "e3", "e4"], [0, 0, 1, 0], [0, 1, 2, 1])
    assert rows == [
        ("e1", "F1", "F1M1"),
        ("e2", "F1", "F1M2"),
        ("e3", "F2", "F2M1"),
        ("e4", "F1", "F1M2"),
    ]


def test_merge_clusters_merges_whole_clusters():
    distance = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.9], [0.9, 0.9, 0.0]])
    assert group.merge_clusters(distance, [0, 1, 2], 0.5) == [0, 0, 1]


def synthetic(endpoint: str, style: str, rng: random.Random, repeats: int = 6) -> list[dict]:
    templates = {
        "plain": ["1, 2, 3, 4, 5", "1, 2, 3, 4, 5", "1, 2, 3, 4, 5", "Here: 1, 2, 3, 4, 5"],
        "dotted": ["1, 2, 3, 4, 5.", "1, 2, 3, 4, 5.", "1, 2, 3, 4, 5.", "1. 1\n2. 2"],
    }
    words = {"plain": "the sea was calm and grey", "dotted": "remember the storm came at night"}
    answers = {f"seq_{k}": [rng.choice(templates[style]) for _ in range(repeats)] for k in range(4)}
    answers["txt_story"] = [" ".join(rng.sample(words[style].split(), 4)) for _ in range(repeats)]
    return rows_for(endpoint, answers)


def test_decide_separates_two_synthetic_models():
    rng = random.Random(3)
    rows = []
    for endpoint in ["ref-01", "ref-02", "ref-03"]:
        rows += synthetic(endpoint, "plain", rng)
    for endpoint in ["ref-04", "ref-05"]:
        rows += synthetic(endpoint, "dotted", rng)
    labels = {endpoint: model for endpoint, _, model in group.decide(rows, factor=3.0)}
    assert labels["ref-01"] == labels["ref-02"] == labels["ref-03"]
    assert labels["ref-04"] == labels["ref-05"]
    assert labels["ref-01"] != labels["ref-04"]


def test_decide_ignores_other_prefixes(tmp_path):
    rng = random.Random(1)
    rows = synthetic("ref-01", "plain", rng) + synthetic("ep-01", "plain", rng)
    rows += synthetic("ref-02", "plain", rng)
    labels = group.decide(rows, prefix="ref-")
    assert [endpoint for endpoint, _, _ in labels] == ["ref-01", "ref-02"]
    out = tmp_path / "x.csv"
    group.write_csv(out, labels)
    assert out.read_text().splitlines()[0] == "endpoint,family,model"
