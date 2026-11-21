import json

import pytest

from euroai.me_query import (
    Job,
    budget_from_models,
    budget_lines,
    classify,
    confirm_spend,
    done_keys,
    is_scored,
    plan_jobs,
    prompt_conflicts,
    read_prompts,
    request_body,
    run_jobs,
    select_endpoints,
    spend_lines,
    unrehearsed_prompts,
    used_from_models,
)

PROMPTS = [("0", "a"), ("1", "b"), ("2", "c")]


def ok_body(text="hi"):
    return {"choices": [{"message": {"content": text}}], "budget": {"used": 1, "limit": 100}}


def test_is_scored():
    assert is_scored("ep-07") and not is_scored("ref-01")


def test_read_prompts_formats(tmp_path):
    txt = tmp_path / "p.txt"
    txt.write_text("first\n\nthird\n")
    assert read_prompts(txt) == [("0", "first"), ("2", "third")]
    csv_file = tmp_path / "p.csv"
    csv_file.write_text('id,prompt\nx,"hello, world"\ny,bye\n')
    assert read_prompts(csv_file) == [("x", "hello, world"), ("y", "bye")]
    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text(json.dumps({"id": 1, "prompt": "p"}) + "\n\n")
    assert read_prompts(jsonl) == [("1", "p")]
    duplicate = tmp_path / "d.csv"
    duplicate.write_text("id,prompt\nx,a\nx,b\n")
    with pytest.raises(ValueError, match="x"):
        read_prompts(duplicate)


def test_done_keys_and_prompt_conflicts():
    rows = [{"endpoint": "ref-01", "id": "0", "repeat": 0, "prompt": "a", "response": "r"}]
    assert done_keys(rows) == {("ref-01", "0", 0)}
    assert prompt_conflicts(rows, [("0", "a")]) == []
    assert prompt_conflicts(rows, [("0", "changed")]) == ["0"]


def test_plan_skips_done_and_respects_budget():
    remaining = {"ref-01": 2, "ref-02": 10}
    done = {("ref-02", "0", 0)}
    jobs, planned, over_budget = plan_jobs(PROMPTS, ["ref-01", "ref-02"], 1, remaining, done)
    assert planned == {"ref-01": 2, "ref-02": 2}
    assert over_budget == 1
    assert ("ref-02", "0", 0) not in {job.key for job in jobs}


def test_plan_round_robin():
    jobs, _, _ = plan_jobs(PROMPTS[:2], ["ref-01", "ref-02"], 2, {"ref-01": 9, "ref-02": 9}, set())
    assert [job.key for job in jobs] == [
        ("ref-01", "0", 0),
        ("ref-02", "0", 0),
        ("ref-01", "1", 0),
        ("ref-02", "1", 0),
        ("ref-01", "0", 1),
        ("ref-02", "0", 1),
        ("ref-01", "1", 1),
        ("ref-02", "1", 1),
    ]


def test_scored_requires_allow():
    remaining = {"ep-01": 100, "ref-01": 100}
    with pytest.raises(ValueError, match="--spend"):
        plan_jobs(PROMPTS, ["ref-01", "ep-01"], 1, remaining, set())
    _, planned, _ = plan_jobs(PROMPTS, ["ep-01"], 1, remaining, set(), allow_scored=True)
    assert planned == {"ep-01": 3}
    with pytest.raises(ValueError, match="ep-99"):
        plan_jobs(PROMPTS, ["ep-99"], 1, remaining, set(), allow_scored=True)


def test_select_endpoints():
    available = ["ep-01", "ep-02", "ref-01", "ref-02"]
    assert select_endpoints("ref", available) == ["ref-01", "ref-02"]
    assert select_endpoints("ep", available) == ["ep-01", "ep-02"]
    assert select_endpoints("all", available) == available
    assert select_endpoints("ref-02, ep-01", available) == ["ref-02", "ep-01"]
    with pytest.raises(ValueError, match="ref-10"):
        select_endpoints("ref-10", available)


def test_spend_lines():
    lines = spend_lines({"ep-01": 3, "ep-02": 0, "ref-01": 5}, {"ep-01": 100, "ep-02": 50})
    text = "\n".join(lines)
    assert "ep-01" in text and "97" in text
    assert "ep-02" not in text and "ref-01" not in text
    assert lines[-1].endswith("3")


def test_confirm_spend():
    assert confirm_spend(0, False, lambda _: "")
    assert not confirm_spend(3, False, lambda _: "3")
    assert confirm_spend(3, True, lambda _: " 3 ")
    assert not confirm_spend(3, True, lambda _: "yes")


def test_classify_rules():
    assert classify(200, {}) == ("ok", 0.0)
    assert classify(429, {"Retry-After": "3"}) == ("wait", 3.0)
    assert classify(429, {"retry-after": "soon"}) == ("wait", 5.0)
    assert classify(429, {}) == ("exhausted", 0.0)
    for status in (502, 503, 504):
        assert classify(status, {}) == ("retry", 15.0)
    assert classify(400, {}) == ("fail", 0.0)


class FakeServer:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def send(self, job):
        self.calls.append(job.key)
        reply = self.replies[job.endpoint]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_run_jobs_exhausted_stops_endpoint():
    jobs = [Job("ep-01", str(i), 0, "p") for i in range(3)]
    jobs += [Job("ep-02", str(i), 0, "p") for i in range(2)]
    server = FakeServer(
        {"ep-01": (429, {}, {"detail": "budget used up"}), "ep-02": (200, {}, ok_body())}
    )
    rows, sleeps, logs = [], [], []
    counts = run_jobs(jobs, server.send, rows.append, sleeps.append, parallel=1, log=logs.append)
    assert [call for call in server.calls if call[0] == "ep-01"] == [("ep-01", "0", 0)]
    assert [row["endpoint"] for row in rows] == ["ep-02", "ep-02"]
    assert counts == {"ok": 2, "failed": 0, "skipped": 2, "exhausted": 1}
    assert any("budget used up" in line for line in logs)


def test_run_jobs_retries_then_writes():
    replies = [
        (503, {}, {"detail": "busy"}),
        (429, {"Retry-After": "2"}, {}),
        ConnectionError("reset"),
        (200, {}, ok_body("answer")),
    ]
    server = FakeServer({"ref-01": replies})
    rows, sleeps = [], []
    counts = run_jobs(
        [Job("ref-01", "0", 0, "p")], server.send, rows.append, sleeps.append, log=print
    )
    assert sleeps == [15.0, 2.0, 30.0]
    assert rows == [
        {
            "endpoint": "ref-01",
            "id": "0",
            "repeat": 0,
            "prompt": "p",
            "response": "answer",
            "budget": {"used": 1, "limit": 100},
        }
    ]
    assert counts["ok"] == 1


def test_run_jobs_gives_up_after_max_retries():
    server = FakeServer({"ref-01": (503, {}, {})})
    sleeps = []
    counts = run_jobs(
        [Job("ref-01", "0", 0, "p")],
        server.send,
        lambda _: None,
        sleeps.append,
        log=lambda _: None,
        max_retries=3,
    )
    assert counts["failed"] == 1 and len(sleeps) == 3


def test_budget_and_request_body():
    body = {"data": [{"id": "ep-01", "budget": {"limit": 100, "used": 7}}]}
    assert budget_from_models(body) == {"ep-01": 93}
    assert request_body(Job("ref-01", "0", 0, "hello")) == {
        "model": "ref-01",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_budget_lines_list_every_endpoint_and_total_the_scored_ones():
    body = {
        "data": [
            {"id": "ep-01", "budget": {"limit": 100, "used": 7}},
            {"id": "ep-02", "budget": {"limit": 100, "used": 100}},
            {"id": "ref-01", "budget": {"limit": 1000000, "used": 12}},
        ]
    }
    assert budget_lines(body) == [
        "ep-01: 93 left of 100",
        "ep-02: 0 left of 100",
        "ref-01: 999988 left of 1000000",
        "scored answers left: 93 of 200",
    ]


def test_used_from_models():
    body = {"data": [{"id": "ep-01", "budget": {"limit": 100, "used": 7}}]}
    assert used_from_models(body) == {"ep-01": 7}


def test_unrehearsed_prompts_need_every_reference():
    prompts = [("a", "first"), ("b", "second")]
    rows = [
        {"endpoint": "ref-01", "id": "a", "repeat": 0},
        {"endpoint": "ref-02", "id": "a", "repeat": 0},
        {"endpoint": "ref-01", "id": "b", "repeat": 0},
        {"endpoint": "ep-01", "id": "b", "repeat": 0},
    ]
    assert unrehearsed_prompts(rows, prompts, ["ref-01", "ref-02"]) == ["b"]
    assert unrehearsed_prompts(rows, prompts, ["ref-01"]) == []
    assert unrehearsed_prompts([], prompts, ["ref-01"]) == ["a", "b"]


def test_answer_row_keeps_the_elapsed_time_when_given():
    from euroai.me_query import Job, answer_row

    job = Job("ep-01", "p1", 0, "hello")
    body = {"choices": [{"message": {"content": "hi"}}], "budget": {"used": 1, "limit": 100}}
    assert "elapsed_s" not in answer_row(job, body)
    assert answer_row(job, {**body, "elapsed_s": 1.25})["elapsed_s"] == 1.25
