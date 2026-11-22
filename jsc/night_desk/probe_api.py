"""One-off look at what the reference endpoints return beyond the answer text.

Reference endpoints only (never scored, no query limit), one request at a time.
uv run --extra api python probe_api.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from euroai.challenge import SERVER_URL, get_task  # noqa: E402
from euroai.credentials import API_KEY_VAR, get_secret  # noqa: E402

BASE_URL = f"{SERVER_URL}/{get_task('model_equality').server_task_id}/v1"
ENDPOINT = "ref-01"
PROMPT = "Name one colour."


def shape(value, depth=0):
    if isinstance(value, dict):
        return {key: shape(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0], depth + 1)] if value else []
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:80] + "..."
    return value


def main() -> int:
    lookup = get_secret(API_KEY_VAR)
    if lookup.value is None:
        print("no key")
        return 1
    session = requests.Session()
    session.headers["X-API-Key"] = lookup.value
    base = {"model": ENDPOINT, "max_tokens": 8, "messages": [{"role": "user", "content": PROMPT}]}
    trials = [
        ("plain", "chat/completions", dict(base)),
        ("logprobs", "chat/completions", {**base, "logprobs": True, "top_logprobs": 5}),
        ("n=2", "chat/completions", {**base, "n": 2}),
        ("raw completions", "completions", {"model": ENDPOINT, "max_tokens": 8, "prompt": "The capital of France is"}),
    ]
    for name, route, body in trials:
        reply = session.post(f"{BASE_URL}/{route}", json=body, timeout=180)
        try:
            payload = reply.json()
        except ValueError:
            payload = {"raw_text": reply.text[:300]}
        print(f"== {name}: HTTP {reply.status_code}")
        print(json.dumps(shape(payload), ensure_ascii=False)[:1500])
    reply = session.get(f"{BASE_URL}/models", timeout=60)
    payload = reply.json()
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    first = entries[0] if isinstance(entries, list) and entries else entries
    print("== models: HTTP", reply.status_code, "| top-level keys:", sorted(payload) if isinstance(payload, dict) else type(payload).__name__)
    print(json.dumps(shape(first), ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
