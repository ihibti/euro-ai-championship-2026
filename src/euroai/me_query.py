"""Guarded querying of the model-equality endpoints: plan within budget, resume, never re-ask.

Follows the organizers' task_template.py: answers are appended as JSON lines
{"endpoint", "id", "repeat", "prompt", "response"}, jobs go round-robin over endpoints, and the
server's status codes mean the same thing here.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORED_PREFIX = "ep-"
REFERENCE_PREFIX = "ref-"
MAX_TOKENS = 256
PARALLEL = 4
RETRY_DELAY = 15.0
NETWORK_DELAY = 30.0
DEFAULT_WAIT = 5.0
MAX_WAIT = 300.0
MAX_RETRIES = 20

Prompt = tuple[str, str]
Key = tuple[str, str, int]


@dataclass(frozen=True)
class Job:
    endpoint: str
    prompt_id: str
    repeat: int
    prompt: str

    @property
    def key(self) -> Key:
        return (self.endpoint, self.prompt_id, self.repeat)


def is_scored(endpoint: str) -> bool:
    return endpoint.startswith(SCORED_PREFIX)


def read_prompts(path: str | Path) -> list[Prompt]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".csv":
        prompts = [(str(row["id"]), row["prompt"]) for row in csv.DictReader(io.StringIO(text))]
    elif path.suffix == ".jsonl":
        items = (json.loads(line) for line in text.splitlines() if line.strip())
        prompts = [(str(item["id"]), item["prompt"]) for item in items]
    else:
        prompts = [
            (str(index), line) for index, line in enumerate(text.split("\n")) if line.strip()
        ]
    seen: set[str] = set()
    for prompt_id, _ in prompts:
        if prompt_id in seen:
            raise ValueError(f"prompt id {prompt_id} appears twice in {path.name}")
        seen.add(prompt_id)
    return prompts


def done_keys(rows: Iterable[Mapping[str, Any]]) -> set[Key]:
    return {(str(row["endpoint"]), str(row["id"]), int(row["repeat"])) for row in rows}


def prompt_conflicts(rows: Iterable[Mapping[str, Any]], prompts: Sequence[Prompt]) -> list[str]:
    current = dict(prompts)
    return sorted(
        {
            str(row["id"])
            for row in rows
            if str(row["id"]) in current and row.get("prompt") != current[str(row["id"])]
        }
    )


def select_endpoints(spec: str, available: Sequence[str]) -> list[str]:
    spec = spec.strip()
    if spec == "ref":
        return [endpoint for endpoint in available if endpoint.startswith(REFERENCE_PREFIX)]
    if spec == "ep":
        return [endpoint for endpoint in available if is_scored(endpoint)]
    if spec == "all":
        return list(available)
    chosen = list(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    unknown = [endpoint for endpoint in chosen if endpoint not in available]
    if unknown:
        raise ValueError(
            f"unknown endpoints: {', '.join(unknown)}; the server lists {', '.join(available)}"
        )
    return chosen


def plan_jobs(
    prompts: Sequence[Prompt],
    endpoints: Sequence[str],
    repeats: int,
    remaining: Mapping[str, int],
    done: set[Key],
    allow_scored: bool = False,
) -> tuple[list[Job], dict[str, int], int]:
    scored = [endpoint for endpoint in endpoints if is_scored(endpoint)]
    if scored and not allow_scored:
        raise ValueError(
            f"scored endpoints {', '.join(scored)} need --spend; every answer there is irreversible"
        )
    unknown = [endpoint for endpoint in endpoints if endpoint not in remaining]
    if unknown:
        raise ValueError(f"unknown endpoints: {', '.join(unknown)}")
    jobs: list[Job] = []
    planned = {endpoint: 0 for endpoint in endpoints}
    over_budget = 0
    for repeat in range(repeats):
        for prompt_id, prompt in prompts:
            for endpoint in endpoints:
                if (endpoint, prompt_id, repeat) in done:
                    continue
                if planned[endpoint] < remaining[endpoint]:
                    planned[endpoint] += 1
                    jobs.append(Job(endpoint, prompt_id, repeat, prompt))
                else:
                    over_budget += 1
    return jobs, planned, over_budget


def scored_total(planned: Mapping[str, int]) -> int:
    return sum(count for endpoint, count in planned.items() if is_scored(endpoint))


def spend_lines(planned: Mapping[str, int], remaining: Mapping[str, int]) -> list[str]:
    lines = [
        f"{endpoint}: {count} answers; {remaining[endpoint]} left now, "
        f"{remaining[endpoint] - count} after"
        for endpoint, count in planned.items()
        if is_scored(endpoint) and count > 0
    ]
    return [*lines, f"total scored answers: {scored_total(planned)}"]


def confirm_spend(total: int, isatty: bool, read_line: Callable[[str], str]) -> bool:
    if total == 0:
        return True
    if not isatty:
        return False
    prompt = f"Type {total} to spend {total} scored answers, anything else cancels: "
    return read_line(prompt).strip() == str(total)


def budget_from_models(body: Mapping[str, Any]) -> dict[str, int]:
    return {
        item["id"]: int(item["budget"]["limit"]) - int(item["budget"]["used"])
        for item in body["data"]
    }


def budget_lines(body: Mapping[str, Any]) -> list[str]:
    limits = {item["id"]: int(item["budget"]["limit"]) for item in body["data"]}
    left = budget_from_models(body)
    scored = [endpoint for endpoint in limits if is_scored(endpoint)]
    return [
        *(f"{endpoint}: {left[endpoint]} left of {limits[endpoint]}" for endpoint in limits),
        f"scored answers left: {sum(left[e] for e in scored)} of {sum(limits[e] for e in scored)}",
    ]


def request_body(job: Job, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    return {
        "model": job.endpoint,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": job.prompt}],
    }


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name.lower()), None)


def classify(status: int, headers: Mapping[str, str]) -> tuple[str, float]:
    if status == 200:
        return "ok", 0.0
    if status == 429:
        retry_after = _header(headers, "Retry-After")
        if retry_after is None:
            return "exhausted", 0.0
        try:
            return "wait", min(max(float(retry_after), 0.0), MAX_WAIT)
        except ValueError:
            return "wait", DEFAULT_WAIT
    if status in (502, 503, 504):
        return "retry", RETRY_DELAY
    return "fail", 0.0


def _detail(body: Any) -> str:
    if isinstance(body, Mapping):
        return str(body.get("detail") or body.get("raw_text") or "")[:300]
    return str(body)[:300]


def answer_row(job: Job, body: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "endpoint": job.endpoint,
        "id": job.prompt_id,
        "repeat": job.repeat,
        "prompt": job.prompt,
    }
    try:
        row["response"] = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        row["response"] = None
        row["raw"] = body
    row["budget"] = body.get("budget") if isinstance(body, Mapping) else None
    return row


def run_jobs(
    jobs: Sequence[Job],
    send: Callable[[Job], tuple[int, Mapping[str, str], Any]],
    write: Callable[[dict[str, Any]], None],
    sleep: Callable[[float], None],
    parallel: int = PARALLEL,
    log: Callable[[str], None] = print,
    max_retries: int = MAX_RETRIES,
) -> dict[str, int]:
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    exhausted: set[str] = set()
    counts = {"ok": 0, "failed": 0, "skipped": 0, "exhausted": 0}

    def finish(name: str, message: str | None = None) -> None:
        with lock:
            counts[name] += 1
            if message:
                log(message)

    def one(job: Job) -> None:
        retries = 0
        while True:
            with lock:
                if job.endpoint in exhausted:
                    counts["skipped"] += 1
                    return
            try:
                status, headers, body = send(job)
            except OSError as exc:
                status, action, delay = None, "retry", NETWORK_DELAY
                detail = f"server unreachable ({type(exc).__name__})"
            else:
                action, delay = classify(status, headers)
                detail = _detail(body)
            if action == "ok":
                row = answer_row(job, body)
                with lock:
                    write(row)
                    counts["ok"] += 1
                    if row["response"] is None:
                        log(f"{job.endpoint} id {job.prompt_id}: unexpected reply, saved raw")
                return
            if action == "exhausted":
                with lock:
                    if job.endpoint in exhausted:
                        counts["skipped"] += 1
                    else:
                        exhausted.add(job.endpoint)
                        counts["exhausted"] += 1
                        log(f"{job.endpoint}: budget used up, skipping it ({detail})")
                return
            if action == "fail":
                finish("failed", f"{job.endpoint} id {job.prompt_id}: HTTP {status}: {detail}")
                return
            if action == "retry":
                retries += 1
                if retries > max_retries:
                    finish("failed", f"{job.endpoint} id {job.prompt_id}: gave up ({detail})")
                    return
                log(f"{job.endpoint}: {detail or status}; retrying in {delay:.0f} s")
            sleep(delay)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(one, jobs))
    return counts
