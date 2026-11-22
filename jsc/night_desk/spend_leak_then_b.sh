#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
S=SCRATCHPAD
cd "$W" || exit 9
free() { while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done; sleep 5; if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi; }
free
echo "leak prompts on the references (3 repeats) start at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/leak.csv --repeats 3 --name night-e 2>&1 | grep -v -E '^(ok|done) ' | tail -3
refs=$(grep -c '"endpoint": "ref-' tasks/model_equality/answers/night-e.jsonl)
echo "reference rows: $refs"
if [ "$refs" -lt 54 ]; then echo "stop: the reference run is incomplete, nothing spent on the scored endpoints"; exit 5; fi
free
echo "leak prompts on the scored endpoints (2 repeats) start at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/leak.csv --repeats 2 --endpoints ep --spend --name night-e 2>&1 | grep -v -E '^(ok|done) ' | tail -3
git add tasks/model_equality/answers/night-e.jsonl tasks/model_equality/probes/leak.csv
git commit -q -m "Record the two leak prompts: 3 repeats on the references, 2 on the 33 scored endpoints" && git push -q origin HEAD
echo "leak answers pushed $(git log --format=%h -1) at $(date +%H:%M:%S); scored rows: $(grep -c '"endpoint": "ep-' tasks/model_equality/answers/night-e.jsonl)"
free
echo "d1 on family B at 20 repeats starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/d1.csv --repeats 20 --endpoints ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -3
git add tasks/model_equality/answers/night-d.jsonl
git commit -q -m "Record the repetition probe on family B at 20 repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
python3 -I "$S/d1_summary.py" tasks/model_equality/answers/night-d.jsonl
