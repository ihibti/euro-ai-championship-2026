#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done
sleep 4
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi
before=$(grep -c '"endpoint": "ep-' tasks/model_equality/answers/round1.jsonl)
echo "identity answers on the scored endpoints start at $(date +%H:%M:%S); scored rows before: $before"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/idn.csv --repeats 10 --endpoints ep --spend --name round1 2>&1 | grep -v -E '^(ok|done) ' | tail -5
after=$(grep -c '"endpoint": "ep-' tasks/model_equality/answers/round1.jsonl)
echo "ended at $(date +%H:%M:%S); scored rows after: $after"
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record identity answers 3 to 10 on the 33 scored endpoints" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1)"
