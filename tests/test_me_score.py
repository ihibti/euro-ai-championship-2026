import subprocess
import sys
from pathlib import Path

import pytest

from euroai.me_score import parse_labels, same_model_mcc

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tasks" / "model_equality" / "upstream"
TRUTH = parse_labels((UPSTREAM / "practice_answers.csv").read_text(), prefix="ref-")


def test_parse_labels_header_optional_and_prefix_filter():
    text = "endpoint,family,model\nref-01,P,S\nep-01,A,A1\n"
    assert parse_labels(text) == {"ref-01": ("P", "S"), "ep-01": ("A", "A1")}
    assert parse_labels(text, prefix="ref-") == {"ref-01": ("P", "S")}
    assert parse_labels("﻿ref-01, P , S\n") == {"ref-01": ("P", "S")}


def test_parse_labels_reads_practice_answers_with_kind_column():
    assert len(TRUTH) == 9
    assert TRUTH["ref-02"] == ("P", "T1")


def test_parse_labels_rejects_short_rows_and_duplicates():
    with pytest.raises(ValueError, match="fields"):
        parse_labels("ref-01,P\n")
    with pytest.raises(ValueError, match="twice"):
        parse_labels("ref-01,P,S\nref-01,P,T\n")


def test_perfect_answer_scores_one():
    result = same_model_mcc(TRUTH, TRUTH)
    assert result.mcc == 1.0
    assert (result.tp, result.fp, result.fn, result.tn) == (10, 0, 0, 26)
    assert result.wrong == []


def test_all_different_scores_zero():
    guess = {endpoint: ("F", endpoint) for endpoint in TRUTH}
    result = same_model_mcc(TRUTH, guess)
    assert result.mcc == 0.0
    assert (result.tp, result.fn) == (0, 10)


def test_negative_correlation_clipped_to_zero():
    truth = {"a": ("F", "1"), "b": ("F", "1"), "c": ("F", "2")}
    guess = {"a": ("F", "1"), "b": ("F", "2"), "c": ("F", "1")}
    assert same_model_mcc(truth, guess).mcc == 0.0


def test_missing_endpoint_raises():
    guess = {endpoint: labels for endpoint, labels in TRUTH.items() if endpoint != "ref-09"}
    with pytest.raises(ValueError, match="ref-09"):
        same_model_mcc(TRUTH, guess)


def test_matches_upstream_score_practice(tmp_path):
    answer = tmp_path / "answer.csv"
    rows = ["ref-01,P,S", "ref-02,P,S", "ref-03,P,S", "ref-04,P,X", "ref-05,P,S"]
    rows += ["ref-06,P,Y", "ref-07,P,X", "ref-08,P,S", "ref-09,P,Z"]
    answer.write_text("endpoint,family,model\n" + "\n".join(rows) + "\n")
    run = subprocess.run(
        [sys.executable, "-I", str(UPSTREAM / "score_practice.py"), str(answer)],
        capture_output=True,
        text=True,
        check=True,
    )
    upstream_mcc = float(run.stdout.split("same model or not): ")[1].split()[0])
    ours = same_model_mcc(TRUTH, parse_labels(answer.read_text(), prefix="ref-"))
    assert upstream_mcc > 0
    assert round(ours.mcc, 3) == upstream_mcc
    assert len(ours.wrong) == ours.fp + ours.fn
