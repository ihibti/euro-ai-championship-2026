import json
import subprocess
from datetime import datetime, timezone

import pytest

from euroai.challenge import CHALLENGE_TASKS, SERVER_URL, get_task
from euroai.provenance import GitState
from euroai.records import (
    board_rows,
    build_result_record,
    format_board,
    git_person,
    load_records,
    record_stem,
    write_record,
)

FIXED = datetime(2026, 11, 21, 16, 30, 0, tzinfo=timezone.utc)
GIT = GitState(commit="abc1234def", branch="main", dirty=False)


def test_registry_matches_task_statements():
    assert set(CHALLENGE_TASKS) == {"watermark_localization", "model_equality", "unlearning"}
    assert get_task("model_equality").server_task_id == "33-model-equality"
    assert get_task("unlearning").server_task_id == "31-unlearning"
    assert get_task("watermark_localization").server_task_id == "34-watermark-localisation-image"
    assert get_task("unlearning").max_bytes == 450_000_000
    assert get_task("watermark_localization").max_bytes == 20_000_000
    assert get_task("model_equality").suffixes == (".csv",)
    assert SERVER_URL == "http://34.147.229.55"


def test_get_task_unknown_lists_names():
    with pytest.raises(ValueError, match="model_equality"):
        get_task("watermark")


def test_record_stem():
    assert record_stem(FIXED, "Ana B", "practice_mcc") == "20261121T163000Z_ana-b_practice-mcc"
    assert record_stem(FIXED, None, "x") == "20261121T163000Z_unknown_x"


def test_build_result_record_fields():
    record = build_result_record(
        task="model_equality",
        metric="practice_mcc",
        score=0.75,
        command="make score",
        notes="n",
        person="ana",
        git=GIT,
        now=FIXED,
    )
    assert record == {
        "task": "model_equality",
        "recorded_at": "2026-11-21T16:30:00+00:00",
        "metric": "practice_mcc",
        "score": 0.75,
        "command": "make score",
        "notes": "n",
        "person": "ana",
        "git": {"commit": "abc1234def", "branch": "main", "dirty": False},
    }


def test_write_record_never_overwrites(tmp_path):
    first = write_record(tmp_path / "results", {"score": 1}, stem="s")
    second = write_record(tmp_path / "results", {"score": 2}, stem="s")
    assert (first.name, second.name) == ("s.json", "s-2.json")
    assert json.loads(first.read_text()) == {"score": 1}
    assert json.loads(second.read_text()) == {"score": 2}


def test_load_records_ignores_non_json(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "notes.txt").write_text("x")
    write_record(tmp_path, {"score": 0.5}, stem="a")
    assert load_records(tmp_path) == [{"score": 0.5, "_file": "a.json"}]


def test_load_records_missing_dir(tmp_path):
    assert load_records(tmp_path / "absent") == []


def test_board_sorts_desc_with_missing_scores_last():
    rows = board_rows([{"score": 0.2}, {"score": None}, {"score": 0.9}, {}])
    assert [row.get("score") for row in rows] == [0.9, 0.2, None, None]


def test_format_board_rows_and_empty():
    record = {
        "score": 0.9,
        "recorded_at": "2026-11-21T16:30:00+00:00",
        "metric": "practice_mcc",
        "person": "ana",
        "git": {"commit": "abc1234def"},
        "notes": "first try",
    }
    lines = format_board(board_rows([record])).splitlines()
    assert lines[0].split()[0] == "score"
    assert "0.9000" in lines[1] and "abc1234" in lines[1] and "first try" in lines[1]
    assert format_board([]) == "no records"


def test_git_person_fake_runner(tmp_path):
    def ok(cmd, **_):
        return subprocess.CompletedProcess(cmd, 0, stdout="Ana B\n", stderr="")

    def failing(cmd, **_):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    assert git_person(tmp_path, run=ok) == "Ana B"
    assert git_person(tmp_path, run=failing) is None
