#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 5; done
echo "channel free at $(date +%H:%M:%S); scored rows in round1.jsonl: $(grep -c '"endpoint": "ep-' tasks/model_equality/answers/round1.jsonl)"
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record the reconnaissance on the 33 scored endpoints: five round-1 prompts at 2 repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1)"
test -f tasks/model_equality/probes/night_d.csv || { echo "stop: night_d.csv is missing"; exit 3; }
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi
echo "night_d on the references starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/night_d.csv --repeats 10 2>&1 | grep -v -E '^(ok|done) ' | tail -5
echo "night_d ended at $(date +%H:%M:%S); rows: $(wc -l < tasks/model_equality/answers/night-d.jsonl)"
git add tasks/model_equality/answers/night-d.jsonl
git commit -q -m "Record night_d on the reference endpoints: the P3 confirmation and two stop-decision variants at 10 repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1)"
