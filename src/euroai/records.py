"""One JSON file per scoring run or submission attempt, and a board over them."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from euroai.provenance import GitState, Runner, slugify

BOARD_COLUMNS = ("score", "recorded_at", "metric", "person", "commit", "notes")


def record_stem(now: datetime, person: str | None, label: str) -> str:
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{slugify(person or 'unknown')}_{slugify(label)}"


def build_result_record(
    *,
    task: str,
    metric: str,
    score: float | None,
    command: str,
    notes: str,
    person: str | None,
    git: GitState,
    now: datetime,
) -> dict[str, Any]:
    return {
        "task": task,
        "recorded_at": now.astimezone(timezone.utc).isoformat(),
        "metric": metric,
        "score": score,
        "command": command,
        "notes": notes,
        "person": person,
        "git": git.to_dict(),
    }


def write_record(directory: str | Path, record: Mapping[str, Any], *, stem: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
    suffix = 1
    while True:
        path = directory / (f"{stem}.json" if suffix == 1 else f"{stem}-{suffix}.json")
        try:
            with path.open("x") as handle:
                handle.write(payload)
            return path
        except FileExistsError:
            suffix += 1


def load_records(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        record["_file"] = path.name
        records.append(record)
    return records


def _has_score(record: Mapping[str, Any]) -> bool:
    score = record.get("score")
    return isinstance(score, int | float) and not isinstance(score, bool)


def board_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(record) for record in records]
    scored = sorted((row for row in rows if _has_score(row)), key=lambda row: -row["score"])
    return scored + [row for row in rows if not _has_score(row)]


def _board_line(row: Mapping[str, Any]) -> str:
    score = f"{row['score']:8.4f}" if _has_score(row) else f"{'-':>8}"
    commit = str((row.get("git") or {}).get("commit") or "-")[:7]
    return "  ".join(
        [
            score,
            f"{str(row.get('recorded_at') or '-'):<25}",
            f"{str(row.get('metric') or '-'):<16}",
            f"{str(row.get('person') or '-'):<12}",
            f"{commit:<7}",
            str(row.get("notes") or ""),
        ]
    ).rstrip()


def format_board(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "no records"
    header = "  ".join(
        [f"{'score':>8}", f"{'recorded_at':<25}", f"{'metric':<16}", f"{'person':<12}"]
        + [f"{'commit':<7}", "notes"]
    )
    return "\n".join([header.lstrip(), *(_board_line(row) for row in rows)])


def git_person(repo_dir: str | Path = ".", run: Runner = subprocess.run) -> str | None:
    try:
        proc = run(
            ["git", "-C", str(repo_dir), "config", "user.name"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = proc.stdout.strip() if proc.returncode == 0 else ""
    return name or None
