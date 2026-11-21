"""A night authorization: what the lead's agent may do on the server while the lead sleeps.

The lead writes the file at a terminal, next to the secrets file and outside every checkout:

{"expires": "2026-11-22T09:00:00+01:00", "answers_per_endpoint": 50,
 "submissions": {"model_equality": 4, "unlearning": 12}}

Agents never create or edit it. Without it, or once it has ended, scored answers and submissions
need a human at a terminal.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

NIGHT_FILE_VAR = "EUROAI_NIGHT_FILE"
DEFAULT_NIGHT_FILE = "~/.config/euroai/night.json"
NIGHT = "night"
HUMAN = "human"


@dataclass(frozen=True)
class NightAuthorization:
    expires: datetime
    answers_per_endpoint: int
    submissions: Mapping[str, int]


def night_file(env: Mapping[str, str]) -> Path:
    return Path(env.get(NIGHT_FILE_VAR) or DEFAULT_NIGHT_FILE).expanduser()


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a whole number, zero or more")
    return value


def parse_authorization(text: str) -> NightAuthorization:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if "expires" not in data:
        raise ValueError("expires is missing")
    try:
        expires = datetime.fromisoformat(str(data["expires"]))
    except ValueError as exc:
        raise ValueError("expires is not an ISO time") from exc
    if expires.tzinfo is None:
        raise ValueError("expires needs a time zone, such as +01:00")
    submissions = data.get("submissions", {})
    if not isinstance(submissions, dict):
        raise ValueError("submissions must map a task to a number")
    return NightAuthorization(
        expires=expires,
        answers_per_endpoint=_count(data.get("answers_per_endpoint", 0), "answers_per_endpoint"),
        submissions={
            str(task): _count(count, f"submissions.{task}") for task, count in submissions.items()
        },
    )


def load_authorization(text: str | None, now: datetime) -> tuple[NightAuthorization | None, str]:
    if text is None:
        return None, "there is no night authorization"
    try:
        night = parse_authorization(text)
    except ValueError as exc:
        return None, f"the night authorization is unreadable: {exc}"
    if now >= night.expires:
        return None, f"the night authorization ended at {night.expires.isoformat()}"
    return night, f"night authorization until {night.expires.isoformat()}"


def submission_refusal(
    night: NightAuthorization, task: str, records: Iterable[Mapping[str, Any]]
) -> str | None:
    allowed = night.submissions.get(task, 0)
    made = sum(1 for record in records if record.get("authorization") == NIGHT)
    if made >= allowed:
        return f"the night authorization allows {allowed} {task} submissions and {made} are made"
    return None


def spend_refusal(
    night: NightAuthorization,
    planned: Mapping[str, int],
    used: Mapping[str, int],
    unrehearsed: Sequence[str],
) -> str | None:
    over = [
        f"{endpoint} ({used[endpoint]} used, {count} planned)"
        for endpoint, count in planned.items()
        if count > 0 and used[endpoint] + count > night.answers_per_endpoint
    ]
    if over:
        return (
            f"the night authorization allows {night.answers_per_endpoint} answers per endpoint; "
            f"over it: {', '.join(over[:5])}"
        )
    if unrehearsed:
        return (
            "these prompts have no saved answer from every reference endpoint yet: "
            f"{', '.join(unrehearsed[:5])}"
        )
    return None


def spend_decision(
    text: str | None,
    now: datetime,
    planned: Mapping[str, int],
    used: Mapping[str, int],
    unrehearsed: Sequence[str],
) -> tuple[bool, str]:
    night, reason = load_authorization(text, now)
    if night is None:
        return False, reason
    refusal = spend_refusal(night, planned, used, unrehearsed)
    if refusal is not None:
        return False, refusal
    return True, reason
