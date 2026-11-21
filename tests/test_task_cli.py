import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from euroai.records import load_records
from euroai.validators import ME_ENDPOINTS

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "sentinel-key-0123456789"
NO_SECRETS = {
    "EUROAI_SECRETS_FILE": "/nonexistent/euroai-test-secrets.env",
    "EUROAI_NIGHT_FILE": "/nonexistent/euroai-test-night.json",
}

pytestmark = pytest.mark.smoke


def run_script(name, *args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**os.environ, **NO_SECRETS, **(env or {})},
        timeout=120,
    )


@pytest.fixture
def answer(tmp_path):
    path = tmp_path / "answer.csv"
    lines = ["endpoint,family,model"] + [f"{e},A,A{int(e[-2:])}" for e in ME_ENDPOINTS]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_submit_without_terminal_refuses(answer):
    run = run_script(
        "submit.py",
        "--task",
        "model_equality",
        "--file",
        str(answer),
        env={"HACKATHON_API_KEY": SENTINEL},
    )
    assert run.returncode == 2, run.stdout + run.stderr
    assert "human at a terminal" in run.stdout or "cool-down" in run.stdout
    assert "Type model_equality" not in run.stdout
    assert SENTINEL not in run.stdout + run.stderr


def test_scripts_run_without_the_lead_files():
    assert set(NO_SECRETS) == {"EUROAI_SECRETS_FILE", "EUROAI_NIGHT_FILE"}
    assert not any(Path(value).exists() for value in NO_SECRETS.values())


def test_submit_without_terminal_leaves_no_record(answer):
    folder = ROOT / "tasks" / "model_equality" / "submissions"
    before = sorted(path.name for path in folder.iterdir())
    run_script(
        "submit.py",
        "--task",
        "model_equality",
        "--file",
        str(answer),
        env={"HACKATHON_API_KEY": SENTINEL},
    )
    assert sorted(path.name for path in folder.iterdir()) == before


def test_submit_check_only(answer):
    run = run_script("submit.py", "--task", "model_equality", "--file", str(answer), "--check")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "not submitted (check only)" in run.stdout


def test_submit_check_rejects_invalid_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ref-01,P,S\n")
    run = run_script("submit.py", "--task", "model_equality", "--file", str(bad), "--check")
    assert run.returncode == 2 and "ref-01" in run.stdout


def test_me_query_without_key_refuses_before_any_network(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello\n")
    env = {"HACKATHON_API_KEY": "", "EUROAI_SECRETS_FILE": str(tmp_path / "absent.env")}
    run = run_script("me_query.py", "--prompts", str(prompts), env=env)
    assert run.returncode == 2, run.stdout + run.stderr
    assert "HACKATHON_API_KEY" in run.stdout and "absent.env" in run.stdout


def test_score_model_equality_prints_one_line_and_records(tmp_path):
    answer = tmp_path / "practice.csv"
    answer.write_text("\n".join(f"ref-0{i},P,M{i}" for i in range(1, 10)) + "\n")
    results = tmp_path / "results"
    run = run_script(
        "score.py",
        "--task",
        "model_equality",
        "--answer",
        str(answer),
        "--notes",
        "all different",
        "--results",
        str(results),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == "score=0.000000\n"
    [record] = load_records(results)
    assert record["score"] == 0.0 and record["metric"] == "practice_mcc"
    assert record["notes"] == "all different"
    assert record["details"]["pairs"]["fn"] == 10
    assert record["command"].startswith("make score TASK=model_equality ANSWER=")


def test_score_watermark_with_synthetic_truth(tmp_path):
    names = np.array(["a", "b"])
    truth_masks = np.zeros((2, 128, 128), np.uint8)
    truth_masks[0, :64] = 1
    truth = tmp_path / "release" / "train" / "masks_128.npz"
    truth.parent.mkdir(parents=True)
    np.savez(truth, names=names, masks=truth_masks)
    (truth.parent / "metadata.json").write_text(
        json.dumps([{"image_name": "a", "method": "pixelseal"}, {"image_name": "b", "method": "x"}])
    )
    answer = tmp_path / "pred.npz"
    np.savez(answer, image_names=names, masks=truth_masks)
    results = tmp_path / "results"
    run = run_script(
        "score.py",
        "--task",
        "watermark_localization",
        "--answer",
        str(answer),
        "--truth",
        str(truth),
        "--results",
        str(results),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == "score=1.000000\n"
    [record] = load_records(results)
    assert record["details"]["by_method"] == {"pixelseal": 1.0, "x": 1.0}


def test_score_other_tasks_exit_nonzero(tmp_path):
    answer = tmp_path / "model.pt"
    answer.write_bytes(b"x")
    run = run_script("score.py", "--task", "unlearning", "--answer", str(answer))
    assert run.returncode == 2 and "tasks/unlearning/README.md" in run.stderr
    assert run.stdout == ""
    missing = run_script(
        "score.py",
        "--task",
        "watermark_localization",
        "--answer",
        str(answer),
        "--truth",
        str(tmp_path / "absent.npz"),
    )
    assert missing.returncode == 2 and "make data" in missing.stderr


def test_board_lists_records(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for index, score in enumerate([0.25, 0.75]):
        record = {"score": score, "metric": "practice_mcc", "notes": f"run {index}"}
        (results / f"r{index}.json").write_text(json.dumps(record))
    run = run_script("board.py", "--task", "model_equality", "--results", str(results))
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert lines[0].startswith("score")
    assert "0.7500" in lines[1] and "0.2500" in lines[2]


def test_data_rejects_unknown_task():
    run = run_script("data.py", "--task", "watermark")
    assert run.returncode == 2 and "invalid choice" in run.stderr
