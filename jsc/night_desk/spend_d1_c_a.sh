#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
S=SCRATCHPAD
cd "$W" || exit 9
while pgrep -f 'spend_leak_then_b.sh' >/dev/null; do sleep 5; done
free() { while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done; sleep 5; if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi; }
free
echo "d1 on family C to 10 repeats starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/d1.csv --repeats 10 --endpoints ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -3
free
echo "d1 on family A to 5 repeats starts at $(date +%H:%M:%S)"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/d1.csv --repeats 5 --endpoints ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -3
git add tasks/model_equality/answers/night-d.jsonl
git commit -q -m "Record the repetition probe on family C at 10 repeats and family A at 5" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
python3 -I "$S/d1_summary.py" tasks/model_equality/answers/night-d.jsonl
