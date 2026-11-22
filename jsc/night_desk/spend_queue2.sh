#!/bin/bash
set -uo pipefail
W=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs
S=SCRATCHPAD
A=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
B=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
C=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
Q="uv run --extra api python scripts/me_query.py"
P=tasks/model_equality/probes
ANS=tasks/model_equality/answers
cd "$W" || exit 9
while pgrep -f 'spend_leak_then_b.sh' >/dev/null; do sleep 5; done
free() { while pgrep -f 'scripts/me_query.py' >/dev/null; do sleep 3; done; sleep 5; if pgrep -f 'scripts/me_query.py' >/dev/null; then echo "stop: another query process started"; exit 4; fi; }
push() { git add "$1"; git commit -q -m "$2" && git push -q origin HEAD; echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S): $2"; }
free
echo "stage 1, leak prompts on family A to 10 repeats, starts at $(date +%H:%M:%S)"
$Q --prompts $P/leak.csv --repeats 10 --endpoints $A --spend --name night-e 2>&1 | grep -v -E '^(ok|done) ' | tail -2
push $ANS/night-e.jsonl "Record the two leak prompts on family A at 10 repeats"
free
for f in $P/bat.csv tasks/model_equality/analysis/modal.py; do
  if [ ! -f "$f" ] || [ -n "$(git status --short -- "$f")" ]; then echo "stop at $(date +%H:%M:%S): $f is missing or not committed; batteries not spent"; exit 5; fi
done
echo "stage 2, batteries on the 33 scored endpoints to 10 repeats, starts at $(date +%H:%M:%S); modal.py frozen at $(git log -1 --format='%h %cd' --date=format:%H:%M:%S -- tasks/model_equality/analysis/modal.py)"
$Q --prompts $P/bat.csv --repeats 10 --endpoints ep --spend --name round1 2>&1 | grep -v -E '^(ok|done) ' | tail -2
push $ANS/round1.jsonl "Record the two batteries on the 33 scored endpoints at 10 repeats"
free
echo "stage 3, leak prompts on families B and C to 6 repeats, starts at $(date +%H:%M:%S)"
$Q --prompts $P/leak.csv --repeats 6 --endpoints $B,$C --spend --name night-e 2>&1 | grep -v -E '^(ok|done) ' | tail -2
push $ANS/night-e.jsonl "Record the two leak prompts on families B and C at 6 repeats"
free
echo "stage 4, d1 on family C to 10 repeats and family A to 5, starts at $(date +%H:%M:%S)"
$Q --prompts $P/d1.csv --repeats 10 --endpoints $C --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -2
free
$Q --prompts $P/d1.csv --repeats 5 --endpoints $A --spend --name night-d 2>&1 | grep -v -E '^(ok|done) ' | tail -2
push $ANS/night-d.jsonl "Record the repetition probe on family C at 10 repeats and family A at 5"
echo "queue ended at $(date +%H:%M:%S)"
