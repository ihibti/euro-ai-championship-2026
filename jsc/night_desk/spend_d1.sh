#!/bin/bash
set -uo pipefail
REPEATS=$1
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
S=SCRATCHPAD
cd "$W" || exit 9
while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done
sleep 6
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi
echo "d1 on the scored endpoints at $REPEATS repeats starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/d1.csv --repeats "$REPEATS" --endpoints ep --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -5
echo "ended at $(date +%H:%M:%S); scored rows in night-d.jsonl: $(grep -c '"endpoint": "ep-' tasks/model_equality/answers/night-d.jsonl)"
git add tasks/model_equality/answers/night-d.jsonl tasks/model_equality/probes/d1.csv
git commit -q -m "Record the repetition probe on the 33 scored endpoints at $REPEATS repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1)"
python3 -I "$S/d1_summary.py" tasks/model_equality/answers/night-d.jsonl
