#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process is running"; exit 4; fi
echo "identity on the 33 scored endpoints to 20 repeats, starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/idn.csv --repeats 20 --endpoints ep --spend --name round1 2>&1 | grep -v -E '^(ok|done) ' | tail -2
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record the identity prompt on the 33 scored endpoints at 20 repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
