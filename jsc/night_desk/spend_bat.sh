#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
while pgrep -f 'spend_leak_then_b.sh' >/dev/null || pgrep -f 'spend_d1_c_a.sh' >/dev/null; do sleep 5; done
free() { while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done; sleep 5; if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi; }
free
for f in tasks/model_equality/probes/bat.csv tasks/model_equality/analysis/modal.py; do
  if [ ! -f "$f" ] || [ -n "$(git status --short -- "$f")" ]; then echo "stop at $(date +%H:%M:%S): $f is missing or not committed; nothing spent"; exit 5; fi
done
echo "modal.py frozen at $(git log -1 --format='%h %cd' --date=format:%H:%M:%S -- tasks/model_equality/analysis/modal.py)"
before=$(grep -c '"endpoint": "ep-' tasks/model_equality/answers/round1.jsonl)
echo "batteries on the scored endpoints to 10 repeats start at $(date +%H:%M:%S); scored rows before: $before"
uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/bat.csv --repeats 10 --endpoints ep --spend --name round1 2>&1 | grep -v -E '^(ok|done) ' | tail -3
echo "ended at $(date +%H:%M:%S); scored rows after: $(grep -c '"endpoint": "ep-' tasks/model_equality/answers/round1.jsonl)"
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record the two batteries on the 33 scored endpoints at 10 repeats" && git push -q origin HEAD
echo "pushed $(git log --format=%h -1)"
