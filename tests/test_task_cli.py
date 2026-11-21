import os
import subprocess
import sys
from pathlib import Path

import pytest

from euroai.validators import ME_ENDPOINTS

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "sentinel-key-0123456789"

pytestmark = pytest.mark.smoke


def run_script(name, *args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=120,
    )


@pytest.fixture
def answer(tmp_path):
    path = tmp_path / "answer.csv"
    lines = ["endpoint,family,model"] + [f"{e},F{e[-2:]},M{e[-2:]}" for e in ME_ENDPOINTS]
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
