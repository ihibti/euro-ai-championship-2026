#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
run() {
  local prompts=$1 repeats=$2 endpoint=$3
  if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process is running"; exit 4; fi
  echo "$prompts on $endpoint to $repeats repeats, starts at $(date +%H:%M:%S)"
  uv run --extra api python scripts/me_query.py --prompts "tasks/model_equality/probes/$prompts" --repeats "$repeats" --name round1 --endpoints "$endpoint" --spend 2>&1 | grep -v -E '^(ok|done) ' | tail -3
}
run bat.csv 16 ep-15
run idn.csv 16 ep-15
run story.csv 16 ep-15
run bat.csv 13 ep-22
run idn.csv 13 ep-22
run story.csv 13 ep-22
git add tasks/model_equality/answers/round1.jsonl
git commit -q -m "Record 24 more round-1 answers on ep-15 and 12 on ep-22 for the registered second read of the fp8 test" -- tasks/model_equality/answers/round1.jsonl && git push -q origin HEAD
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
uv run -q python scripts/me_query.py --budget 2>&1 | grep -E "ep-(15|22):"
