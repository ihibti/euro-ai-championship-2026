"""Query model-equality endpoints within budget, resuming without asking anything twice.

uv run --extra api python scripts/me_query.py --prompts prompts.txt
uv run --extra api python scripts/me_query.py --prompts prompts.txt --endpoints ep --spend

The first form uses the free ref-* endpoints. Scored ep-* endpoints need --spend, and you must
type the total number of scored answers at a terminal. Answers are appended to
tasks/model_equality/answers/<name>.jsonl; commit that file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import SERVER_URL, get_task  # noqa: E402
from euroai.credentials import API_KEY_VAR, get_secret, missing_secret_message  # noqa: E402
from euroai.me_query import (  # noqa: E402
    Job,
    budget_from_models,
    confirm_spend,
    done_keys,
    plan_jobs,
    prompt_conflicts,
    read_prompts,
    request_body,
    run_jobs,
    scored_total,
    select_endpoints,
    spend_lines,
)
from euroai.provenance import slugify  # noqa: E402
from euroai.tracking import read_jsonl  # noqa: E402

ANSWERS_DIR = ROOT / "tasks" / "model_equality" / "answers"
BASE_URL = f"{SERVER_URL}/{get_task('model_equality').server_task_id}/v1"


def make_send(session: Any):
    def send(job: Job) -> tuple[int, Any, Any]:
        reply = session.post(f"{BASE_URL}/chat/completions", json=request_body(job), timeout=180)
        try:
            body = reply.json()
        except ValueError:
            body = {"raw_text": reply.text[:300]}
        return reply.status_code, reply.headers, body

    return send


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prompts", required=True, type=Path, help=".txt, .csv (id,prompt), .jsonl"
    )
    parser.add_argument("--name", help="answers file name; defaults to the prompts file name")
    parser.add_argument("--endpoints", default="ref", help="ref, ep, all, or a comma list")
    parser.add_argument("--repeats", type=int, default=1, help="answers per prompt and endpoint")
    parser.add_argument("--spend", action="store_true", help="allow scored ep-* endpoints")
    args = parser.parse_args(argv)
    lookup = get_secret(API_KEY_VAR)
    if lookup.value is None:
        print(f"refused: {missing_secret_message(lookup)}")
        return 2
    prompts = read_prompts(args.prompts)
    answers = ANSWERS_DIR / f"{slugify(args.name or args.prompts.stem)}.jsonl"
    rows = read_jsonl(answers) if answers.exists() else []
    conflicts = prompt_conflicts(rows, prompts)
    if conflicts:
        print(f"refused: {answers} holds other prompts under ids {', '.join(conflicts[:5])}")
        print("use another --name for a different prompt set")
        return 2
    import requests

    session = requests.Session()
    session.headers["X-API-Key"] = lookup.value
    reply = session.get(f"{BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        print(f"error: HTTP {reply.status_code} from /models")
        return 1
    remaining = budget_from_models(reply.json())
    done = done_keys(rows)
    try:
        endpoints = select_endpoints(args.endpoints, list(remaining))
        jobs, planned, over_budget = plan_jobs(
            prompts, endpoints, args.repeats, remaining, done, allow_scored=args.spend
        )
    except ValueError as exc:
        print(f"refused: {exc}")
        return 2
    print(
        f"{len(prompts)} prompts x {len(endpoints)} endpoints x {args.repeats} repeats: "
        f"{len(done)} answers already in {answers.name}, {len(jobs)} to query"
        + (f", {over_budget} left out (over budget)" if over_budget else "")
    )
    total = scored_total(planned)
    if total:
        for line in spend_lines(planned, remaining):
            print(line)
        human = sys.stdin.isatty() and sys.stdout.isatty()
        if not confirm_spend(total, human, input):
            print("refused: scored answers need a human at a terminal typing the total")
            return 2
    if not jobs:
        print("nothing to query")
        return 0
    answers.parent.mkdir(parents=True, exist_ok=True)
    with answers.open("a", encoding="utf-8") as handle:

        def write(row: dict[str, Any]) -> None:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        counts = run_jobs(jobs, make_send(session), write, time.sleep, log=print)
    print(f"finished: {counts}. Commit {answers.relative_to(ROOT)}.")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
