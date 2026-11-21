"""The only path from a local file to a submission.

Order: server id known, file valid, cool-down over, a human at a terminal, API key present,
summary, typed task name, upload, record. Every attempt that reaches the upload is recorded,
including failures, because the server counts failed submissions toward its cool-down.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from euroai.challenge import SERVER_URL, TaskInfo, get_task
from euroai.credentials import (
    API_KEY_VAR,
    FileReader,
    SecretLookup,
    find_secret,
    missing_secret_message,
    read_text_or_none,
)
from euroai.provenance import GitState, sha256_file
from euroai.records import load_records, record_stem, write_record
from euroai.validators import ValidationReport, validate_file

COOLDOWN = timedelta(minutes=30)
SUBMITTED = 0
FAILED = 1
REFUSED = 2

Poster = Callable[[str, Mapping[str, str], Path], tuple[int, Any]]
Validator = Callable[[str, Path], ValidationReport]


@dataclass(frozen=True)
class SubmitDeps:
    post: Poster
    isatty: Callable[[], bool]
    read_line: Callable[[str], str]
    now: Callable[[], datetime]
    env: Mapping[str, str]
    out: Callable[[str], None]
    git: Callable[[], GitState]
    person: Callable[[], str | None]
    read_file: FileReader = read_text_or_none
    validate: Validator = validate_file


def _parse_time(value: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def latest_attempt(records: Iterable[Mapping[str, Any]]) -> datetime | None:
    times = [_parse_time(record["attempted_at"]) for record in records if "attempted_at" in record]
    return max((moment for moment in times if moment is not None), default=None)


def cooldown_left(last: datetime | None, now: datetime) -> timedelta:
    if last is None:
        return timedelta(0)
    return max(timedelta(0), last + COOLDOWN - now)


def _detail(body: Any) -> str:
    if isinstance(body, Mapping):
        for key in ("detail", "message", "raw_text"):
            if body.get(key):
                return str(body[key])[:500]
        return json.dumps(body, default=str)[:500]
    return str(body)[:500]


def interpret_response(status_code: int, body: Any) -> tuple[str, float | None, str]:
    if status_code == 200 and isinstance(body, Mapping) and body.get("status") == "success":
        raw = body.get("score")
        score = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
        parts = [
            f"score {raw}",
            f"previous best {body.get('previous_best')}",
            f"improved {body.get('improved')}",
        ]
        if body.get("message"):
            parts.append(str(body["message"]))
        return "success", score, "; ".join(parts)
    return "error", None, f"HTTP {status_code}: {_detail(body)}"


def _scrub(text: str, secret: str) -> str:
    for form in {secret, json.dumps(secret)[1:-1]}:
        text = text.replace(form, "***")
    return text


def _wait_text(left: timedelta) -> str:
    seconds = int(left.total_seconds())
    return f"{seconds // 60} min {seconds % 60} s"


def _summary(
    task: str,
    info: TaskInfo,
    path: Path,
    report: ValidationReport,
    git: GitState,
    last: datetime | None,
    lookup: SecretLookup,
) -> list[str]:
    commit = (git.commit or "unknown")[:12] + (" with uncommitted changes" if git.dirty else "")
    checks = "; ".join(["passed", *report.notes])
    return [
        f"task:      {task} (server id {info.server_task_id})",
        f"server:    {SERVER_URL}/submit/{info.server_task_id}",
        f"file:      {path} ({path.stat().st_size} bytes, sha256 {sha256_file(path)[:16]}...)",
        f"commit:    {commit}",
        f"checks:    {checks}",
        f"last try:  {last.isoformat() if last else 'none recorded'}",
        f"api key:   {API_KEY_VAR} from the {lookup.source}",
        "This uses the task's submission slot for the next 30 minutes, even if it fails.",
    ]


def _report_errors(deps: SubmitDeps, task: str, path: Path, report: ValidationReport) -> None:
    deps.out(f"refused: {path} fails the {task} checks:")
    for error in report.errors:
        deps.out(f"  - {error}")


def run_submission(
    task: str,
    path: str | Path,
    records_dir: str | Path,
    deps: SubmitDeps,
    *,
    check_only: bool = False,
) -> int:
    info = get_task(task)
    path = Path(path)
    if info.server_task_id is None:
        deps.out(f"refused: the server task id for {task} is not known")
        return REFUSED
    report = deps.validate(task, path)
    if not report.ok:
        _report_errors(deps, task, path, report)
        return REFUSED
    last = latest_attempt(load_records(records_dir))
    left = cooldown_left(last, deps.now())
    if check_only:
        deps.out(f"checks passed for {path}: {'; '.join(report.notes) or 'ok'}")
        wait = f"cool-down {_wait_text(left)} left" if left else "no cool-down"
        deps.out(f"not submitted (check only); {wait}")
        return SUBMITTED
    if left > timedelta(0):
        deps.out(
            f"refused: cool-down, the last {task} attempt was at {last.isoformat()}; "
            f"wait {_wait_text(left)}. Pull first to see teammates' attempts."
        )
        return REFUSED
    if not deps.isatty():
        deps.out("refused: submitting needs a human at a terminal; agents and scripts may not")
        return REFUSED
    lookup = find_secret(API_KEY_VAR, deps.env, deps.read_file)
    if lookup.value is None:
        deps.out(f"refused: {missing_secret_message(lookup)}")
        return REFUSED
    key = lookup.value

    def say(line: str) -> None:
        deps.out(_scrub(line, key))

    for line in _summary(task, info, path, report, deps.git(), last, lookup):
        say(line)
    if deps.read_line(f"Type {task} to submit, anything else cancels: ").strip() != task:
        say("cancelled: nothing was sent")
        return REFUSED
    attempted_at = deps.now()
    url = f"{SERVER_URL}/submit/{info.server_task_id}"
    try:
        status_code, body = deps.post(url, {"X-API-Key": key}, path)
        status, score, message = interpret_response(status_code, body)
    except OSError as exc:
        status_code, body = None, None
        status, score, message = "error", None, f"{type(exc).__name__}: {exc}"
    person = deps.person()
    record = {
        "task": task,
        "server_task_id": info.server_task_id,
        "attempted_at": attempted_at.astimezone(timezone.utc).isoformat(),
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "validation_notes": report.notes,
        "git": deps.git().to_dict(),
        "person": person,
        "status": status,
        "http_status": status_code,
        "score": score,
        "message": message,
        "response": body,
    }
    scrubbed = json.loads(_scrub(json.dumps(record, default=str), key))
    written = write_record(records_dir, scrubbed, stem=record_stem(attempted_at, person, task))
    say(f"{status}: {message}")
    say(f"recorded {written}; commit and push it now so teammates see the cool-down")
    return SUBMITTED if status == "success" else FAILED
