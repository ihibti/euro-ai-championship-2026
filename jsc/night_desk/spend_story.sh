#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
Q="uv run --extra api python scripts/me_query.py --prompts tasks/model_equality/probes/story.csv --repeats 10 --name round1"
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process is running"; exit 4; fi
echo "story prompt on the nine references to 10 repeats, starts at $(date +%H:%M:%S)"
$Q --endpoints ref 2>&1 | grep -v -E '^(ok|done) ' | tail -2
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record the story prompt on the nine references at 10 repeats" -- tasks/model_equality/answers/round1.jsonl && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process is running"; exit 4; fi
echo "story prompt on the 33 scored endpoints to 10 repeats, starts at $(date +%H:%M:%S)"
$Q --endpoints ep --spend 2>&1 | grep -v -E '^(ok|done) ' | tail -2
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record the story prompt on the 33 scored endpoints at 10 repeats" -- tasks/model_equality/answers/round1.jsonl && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
