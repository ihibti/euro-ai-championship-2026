#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
cd "$W" || exit 9
run() {
  local prompts=$1 repeats=$2 name=$3 label=$4
  if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process is running"; exit 4; fi
  echo "$label on ref-01 and ref-02 to $repeats repeats, starts at $(date +%H:%M:%S)"
  uv run --extra api python scripts/me_query.py --prompts "tasks/model_equality/probes/$prompts" --repeats "$repeats" --name "$name" --endpoints ref-01,ref-02 2>&1 | grep -v -E '^(ok|done) ' | tail -2
  git add "tasks/model_equality/answers/$name.jsonl" "tasks/model_equality/probes/$prompts"
  git commit -q -m "Record $label on the plain and fp8 references at $repeats repeats" -- "tasks/model_equality/answers/$name.jsonl" "tasks/model_equality/probes/$prompts" && git push -q origin HEAD
  echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
}
run p3.csv 30 night-b "the repetition probe p3"
run fp8_hint.csv 14 round1 "the trivia battery and identity"
echo "HINT ROWS COMPLETE at $(date +%H:%M:%S)"
run night_d.csv 30 night-d "the stop probes"
run round1.csv 14 round1 "round 1"
