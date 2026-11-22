"""Tests for within.py. Run with: uv run pytest tasks/model_equality/analysis"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import within  # noqa: E402


def test_parse_probes_defaults_to_the_flag_kind_and_checks_counters():
    probes = within.parse_probes("d1_repeat150:count, d4_copycont,idn_identity:identity")
    assert probes == [
        within.Probe("d1_repeat150", "count"),
        within.Probe("d4_copycont", "flag"),
        within.Probe("idn_identity", "identity"),
    ]
    with pytest.raises(ValueError):
        within.parse_probes("d4_copycont:count")
    with pytest.raises(ValueError):
        within.parse_probes("d1_repeat150:size")


def copies(n: int, hits: int, total: int = 10) -> dict[str, list[bool]]:
    return {f"s{k}": [True] * hits + [False] * (total - hits) for k in range(n)}


def test_partition_keeps_the_majority_and_sets_apart_a_runaway():
    flags = {**copies(5, 2), "lora": [True] * 10}
    parts = within.partition(sorted(flags), [("flag", flags)])
    assert parts == [["s0", "s1", "s2", "s3", "s4"], ["lora"]]


def test_two_set_apart_endpoints_share_a_model_unless_a_pairwise_test_separates_them():
    flags = {**copies(6, 1, 20), "x": [True] * 19 + [False], "y": [True] * 20}
    together = within.partition(sorted(flags), [("flag", flags)])
    assert together[-1] == ["x", "y"] or together[-1] == ["y", "x"]
    counts = {f"s{k}": [100 + k, 110, 120, 130, 125, 115] for k in range(6)}
    counts["x"] = [300, 305, 310, 302, 299, 301]
    counts["y"] = [600, 610, 605, 620, 615, 601]
    parts = within.partition(sorted(counts), [("count", counts)])
    assert ["x"] in parts and ["y"] in parts
    assert parts[0] == [f"s{k}" for k in range(6)]


def test_grouping_reads_rows_per_family_and_labels_rows_in_submission_format():
    rows = [
        {"endpoint": f"ref-0{k}", "id": "d1_repeat150", "repeat": r, "response": "river"}
        for k in range(1, 7)
        for r in range(8)
    ] + [
        {"endpoint": "ref-07", "id": "d1_repeat150", "repeat": r, "response": "river " * 256}
        for r in range(8)
    ]
    families = {"P": [f"ref-0{k}" for k in range(1, 8)]}
    parts = within.grouping(rows, families, [within.Probe("d1_repeat150", "flag")])
    assert parts == {"P": [[f"ref-0{k}" for k in range(1, 7)], ["ref-07"]]}
    labels = within.label_rows(parts)
    assert labels[0] == ("ref-01", "A", "A1")
    assert labels[-1] == ("ref-07", "A", "A2")


def test_identity_flags_read_the_class_against_the_family_base_group():
    rows = [
        {"endpoint": "e1", "id": "idn", "repeat": 0, "response": "I am Qwen, by Alibaba."},
        {"endpoint": "e2", "id": "idn", "repeat": 0, "response": "I am Qwen, by Alibaba."},
        {"endpoint": "e3", "id": "idn", "repeat": 0, "response": "My name is Emily Chen."},
    ]
    flags = within.identity_flags(rows, "idn", ["e1", "e2", "e3"])
    assert flags == {"e1": [False], "e2": [False], "e3": [True]}


def test_resample_keeps_the_number_of_answers_per_cell():
    rows = [{"endpoint": "e", "id": "p", "repeat": r, "response": str(r)} for r in range(5)]
    rows.append({"endpoint": "f", "id": "p", "repeat": 0, "response": "x"})
    out = within.resample(rows, random.Random(0))
    assert sum(r["endpoint"] == "e" for r in out) == 5
    assert sum(r["endpoint"] == "f" for r in out) == 1
    assert {r["response"] for r in out if r["endpoint"] == "e"} <= {"0", "1", "2", "3", "4"}


def test_model_mcc_scores_pairs_against_the_truth():
    truth = {"a": ("P", "S"), "b": ("P", "S"), "c": ("P", "T")}
    assert within.model_mcc(truth, [("a", "A", "A1"), ("b", "A", "A1"), ("c", "A", "A2")]) == 1.0
    assert within.model_mcc(truth, [("a", "A", "A1"), ("b", "A", "A1"), ("c", "A", "A1")]) == 0.0
