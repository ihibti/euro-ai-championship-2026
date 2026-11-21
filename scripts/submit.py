"""Submit one file to the hackathon server, through every guard in euroai.gate.

make submit TASK=model_equality FILE=path/to/answer.csv
make check  TASK=model_equality FILE=path/to/answer.csv    validate only; agents may run this
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import CHALLENGE_TASKS  # noqa: E402
from euroai.credentials import permission_warning, secrets_file  # noqa: E402
from euroai.gate import SubmitDeps, run_submission  # noqa: E402
from euroai.provenance import git_state  # noqa: E402
from euroai.records import git_person  # noqa: E402
from euroai.validators import ValidationReport, validate_file  # noqa: E402
from euroai.wm_score import TEST_METADATA, read_metadata  # noqa: E402

UPLOAD_TIMEOUT = (30, 600)


def http_post(url: str, headers: Mapping[str, str], path: Path) -> tuple[int, Any]:
    import requests

    with path.open("rb") as handle:
        response = requests.post(
            url,
            headers=dict(headers),
            files={"file": (path.name, handle, "application/octet-stream")},
            timeout=UPLOAD_TIMEOUT,
        )
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text[:2000]}
    return response.status_code, body


def watermark_test_names(task: str) -> Collection[str] | None:
    path = ROOT / TEST_METADATA
    if task != "watermark_localization" or not path.is_file():
        return None
    return list(read_metadata(path.read_text()))


def validate(task: str, path: Path) -> ValidationReport:
    return validate_file(task, path, expected_names=watermark_test_names(task))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(CHALLENGE_TASKS))
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="validate only; never submit")
    args = parser.parse_args(argv)
    human = sys.stdin.isatty() and sys.stdout.isatty()
    if not args.check and human and importlib.util.find_spec("requests") is None:
        print("refused: requests is not installed; run through make submit (uv run --extra api)")
        return 2
    secrets = secrets_file(os.environ)
    if not args.check and secrets.is_file():
        warning = permission_warning(secrets, secrets.stat().st_mode)
        if warning:
            print(warning)
    deps = SubmitDeps(
        post=http_post,
        isatty=lambda: human,
        read_line=input,
        now=lambda: datetime.now(timezone.utc),
        env=os.environ,
        out=print,
        git=lambda: git_state(ROOT),
        person=lambda: git_person(ROOT),
        validate=validate,
    )
    records = ROOT / "tasks" / args.task / "submissions"
    return run_submission(args.task, args.file, records, deps, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
