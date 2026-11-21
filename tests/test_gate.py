import dataclasses
import json
from datetime import datetime, timedelta, timezone

import pytest

from euroai import challenge
from euroai.gate import (
    COOLDOWN,
    SubmitDeps,
    cooldown_left,
    interpret_response,
    latest_attempt,
    run_submission,
)
from euroai.provenance import GitState, sha256_file
from euroai.records import load_records
from euroai.validators import ME_ENDPOINTS

KEY = "sk-team-key-do-not-leak"
NOW = datetime(2026, 11, 21, 18, 0, 0, tzinfo=timezone.utc)
SUCCESS = {"status": "success", "score": 0.42, "previous_best": 0.3, "improved": True}


class Harness:
    def __init__(
        self, tmp_path, *, tty=True, typed="model_equality", response=(200, SUCCESS), night=None
    ):
        self.records = tmp_path / "submissions"
        self.lines = []
        self.posts = []
        self.prompts = []
        self.response = response
        self.now = NOW
        self.deps = SubmitDeps(
            post=self.post,
            isatty=lambda: tty,
            read_line=self.read_line(typed),
            now=lambda: self.now,
            env={"HACKATHON_API_KEY": KEY},
            out=self.lines.append,
            git=lambda: GitState(commit="abc1234def", branch="main", dirty=False),
            person=lambda: "ana",
            read_file=lambda path: night if path.name == "night.json" else None,
        )

    def post(self, url, headers, path):
        self.posts.append((url, dict(headers), path))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def read_line(self, typed):
        def read(prompt):
            self.prompts.append(prompt)
            return typed

        return read

    def output(self):
        return "\n".join(self.lines)

    def run(self, path, task="model_equality"):
        return run_submission(task, path, self.records, self.deps)


@pytest.fixture
def answer(tmp_path):
    path = tmp_path / "answer.csv"
    lines = ["endpoint,family,model"] + [f"{e},F{e[-2:]},M{e[-2:]}" for e in ME_ENDPOINTS]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cooldown_helpers():
    assert COOLDOWN.total_seconds() == 30 * 60
    assert cooldown_left(None, NOW) == timedelta(0)
    assert cooldown_left(NOW - timedelta(minutes=10), NOW) == timedelta(minutes=20)
    assert cooldown_left(NOW - timedelta(minutes=45), NOW) == timedelta(0)
    records = [
        {"attempted_at": "2026-11-21T17:00:00+00:00"},
        {"attempted_at": "2026-11-21T17:40:00+00:00"},
        {"attempted_at": "garbage"},
        {},
    ]
    assert latest_attempt(records) == datetime(2026, 11, 21, 17, 40, tzinfo=timezone.utc)
    assert latest_attempt([]) is None


def test_interpret_response():
    status, score, message = interpret_response(200, SUCCESS)
    assert (status, score) == ("success", 0.42)
    assert "0.42" in message and "0.3" in message
    assert interpret_response(429, {"detail": "wait 12 minutes"}) == (
        "error",
        None,
        "HTTP 429: wait 12 minutes",
    )
    assert interpret_response(200, {"status": "failed", "message": "bad file"})[0] == "error"
    assert interpret_response(502, {"raw_text": "<html>"})[2] == "HTTP 502: <html>"


def test_invalid_file_never_posts(tmp_path):
    harness = Harness(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("endpoint,family,model\nep-01,A,A1\n")
    assert harness.run(bad) == 2
    assert harness.posts == [] and harness.prompts == []
    assert load_records(harness.records) == []
    assert "missing endpoints" in harness.output()


def test_cooldown_blocks_including_errors(tmp_path, answer):
    harness = Harness(tmp_path, response=(500, {"detail": "boom"}))
    assert harness.run(answer) == 1
    assert load_records(harness.records)[0]["status"] == "error"
    harness.now = NOW + timedelta(minutes=29)
    assert harness.run(answer) == 2
    assert len(harness.posts) == 1
    assert "cool-down" in harness.output()
    harness.now = NOW + timedelta(minutes=31)
    harness.response = (200, SUCCESS)
    assert harness.run(answer) == 0
    assert len(harness.posts) == 2


def test_network_error_is_recorded_and_blocks(tmp_path, answer):
    harness = Harness(tmp_path, response=ConnectionError("connection reset"))
    assert harness.run(answer) == 1
    record = load_records(harness.records)[0]
    assert record["status"] == "error" and "connection reset" in record["message"]
    harness.now = NOW + timedelta(minutes=5)
    assert harness.run(answer) == 2
    assert len(harness.posts) == 1


def test_non_tty_refuses(tmp_path, answer):
    harness = Harness(tmp_path, tty=False)
    assert harness.run(answer) == 2
    assert harness.posts == [] and harness.prompts == []
    assert "terminal" in harness.output()


def test_missing_api_key_refuses(tmp_path, answer):
    harness = Harness(tmp_path)
    harness.deps = dataclasses.replace(
        harness.deps, env={"EUROAI_SECRETS_FILE": str(tmp_path / "none.env")}
    )
    assert harness.run(answer) == 2
    assert harness.posts == []
    assert "HACKATHON_API_KEY" in harness.output() and "none.env" in harness.output()


def test_wrong_confirmation_refuses(tmp_path, answer):
    harness = Harness(tmp_path, typed="yes")
    assert harness.run(answer) == 2
    assert harness.posts == []
    assert load_records(harness.records) == []
    assert harness.prompts and "model_equality" in harness.prompts[0]


def test_success_records_score_and_hash(tmp_path, answer):
    harness = Harness(tmp_path)
    assert harness.run(answer) == 0
    url, headers, path = harness.posts[0]
    assert url == "http://34.147.229.55/submit/33-model-equality"
    assert headers == {"X-API-Key": KEY} and path == answer
    [record] = load_records(harness.records)
    assert record["status"] == "success" and record["score"] == 0.42
    assert record["sha256"] == sha256_file(answer)
    assert record["git"]["commit"] == "abc1234def" and record["person"] == "ana"
    assert record["attempted_at"] == NOW.isoformat()
    assert "0.42" in harness.output()
    assert "commit and push" in harness.output()


def test_api_key_never_in_output_or_record(tmp_path, answer):
    harness = Harness(tmp_path, response=(401, {"detail": f"invalid key {KEY}"}))
    harness.run(answer)
    harness.now = NOW + timedelta(hours=1)
    harness.response = OSError(f"proxy refused header X-API-Key: {KEY}")
    harness.run(answer)
    assert len(harness.posts) == 2
    assert KEY not in harness.output()
    for path in harness.records.glob("*.json"):
        assert KEY not in path.read_text()
    assert all(json.loads(p.read_text())["status"] == "error" for p in harness.records.glob("*"))


def test_task_without_server_id_refuses(tmp_path, answer, monkeypatch):
    info = challenge.CHALLENGE_TASKS["model_equality"]
    monkeypatch.setitem(
        challenge.CHALLENGE_TASKS, "model_equality", dataclasses.replace(info, server_task_id=None)
    )
    harness = Harness(tmp_path)
    assert harness.run(answer) == 2
    assert harness.posts == []


def test_check_only_validates_without_tty_key_or_post(tmp_path, answer):
    harness = Harness(tmp_path, tty=False)
    harness.deps = dataclasses.replace(harness.deps, env={})
    assert (
        run_submission("model_equality", answer, harness.records, harness.deps, check_only=True)
        == 0
    )
    assert harness.posts == [] and harness.prompts == []
    assert "not submitted" in harness.output() and "33 endpoints" in harness.output()


def test_interrupted_upload_is_recorded_and_blocks(tmp_path, answer):
    harness = Harness(tmp_path, response=KeyboardInterrupt())
    assert harness.run(answer) == 1
    [record] = load_records(harness.records)
    assert record["status"] == "error" and "interrupted" in record["message"]
    harness.now = NOW + timedelta(minutes=1)
    assert harness.run(answer) == 2


NIGHT_TEXT = (
    '{"expires": "2026-11-22T09:00:00+01:00", "answers_per_endpoint": 50, '
    '"submissions": {"model_equality": 2}}'
)


def test_human_submission_is_recorded_as_human(tmp_path, answer):
    harness = Harness(tmp_path)
    assert harness.run(answer) == 0
    [record] = load_records(harness.records)
    assert record["authorization"] == "human"


def test_night_authorization_submits_without_a_typed_confirmation(tmp_path, answer):
    harness = Harness(tmp_path, tty=False, night=NIGHT_TEXT)
    assert harness.run(answer) == 0
    assert len(harness.posts) == 1 and harness.prompts == []
    [record] = load_records(harness.records)
    assert record["authorization"] == "night" and record["score"] == 0.42
    assert "night authorization until" in harness.output()
    assert KEY not in harness.output()


def test_night_authorization_keeps_cooldown_and_cap(tmp_path, answer):
    harness = Harness(tmp_path, tty=False, night=NIGHT_TEXT)
    assert harness.run(answer) == 0
    assert harness.run(answer) == 2 and "cool-down" in harness.output()
    harness.now = NOW + COOLDOWN
    assert harness.run(answer) == 0
    harness.now = NOW + 2 * COOLDOWN
    assert harness.run(answer) == 2
    assert len(harness.posts) == 2
    assert "allows 2 model_equality submissions" in harness.output()


def test_night_authorization_expired_unreadable_or_other_task(tmp_path, answer):
    late = Harness(tmp_path / "late", tty=False, night=NIGHT_TEXT)
    late.now = datetime(2026, 11, 22, 8, 0, 0, tzinfo=timezone.utc)
    assert late.run(answer) == 2 and "ended" in late.output() and late.posts == []
    broken = Harness(tmp_path / "broken", tty=False, night="{")
    assert broken.run(answer) == 2 and "unreadable" in broken.output() and broken.posts == []
    none = '{"expires": "2026-11-22T09:00:00+01:00", "submissions": {"unlearning": 3}}'
    other = Harness(tmp_path / "other", tty=False, night=none)
    assert other.run(answer) == 2 and other.posts == []
    assert "allows 0 model_equality submissions" in other.output()


def test_a_human_at_a_terminal_ignores_the_night_authorization(tmp_path, answer):
    harness = Harness(tmp_path, typed="no", night=NIGHT_TEXT)
    assert harness.run(answer) == 2
    assert harness.posts == [] and len(harness.prompts) == 1
