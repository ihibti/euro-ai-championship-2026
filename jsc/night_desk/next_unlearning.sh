#!/bin/bash
set -uo pipefail
G=$HOME/orca/workspaces/euro-ai-championship-2026/brill
R=/p/scratch/training2666/hibti1/Begg/unlearning/output/kkrgp
fetch_at="$1"
submit_at="$2"
now() { echo "$((10#$(date +%H%M%S)))"; }
cd "$G" || exit 9
until [ "$(now)" -ge "$((10#$fetch_at))" ]; do sleep 2; done
git fetch -q origin || { echo "stop: git fetch failed"; exit 5; }
pick=$(git show origin/kkrgp/unlearning-first-moves:tasks/unlearning/submit_queue.md | python3 -I -c '
import json, pathlib, re, sys
done = set()
for path in pathlib.Path("tasks/unlearning/submissions").glob("*.json"):
    done.add(json.loads(path.read_text()).get("sha256", ""))
for line in sys.stdin:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) >= 2 and re.fullmatch(r"[0-9a-f]{64}", cells[1]) and re.fullmatch(r"[A-Za-z0-9_.-]+\.pt", cells[0]):
        if cells[1] not in done:
            print(cells[0], cells[1])
            break
')
if [ -z "$pick" ]; then echo "stop: the queue has no line without a record"; exit 3; fi
name=${pick% *}; want=${pick#* }
echo "queue head at $(date +%H:%M:%S): $name $want (queue commit $(git rev-parse --short origin/kkrgp/unlearning-first-moves))"
file="data/unlearning/candidates/$name"
if [ ! -f "$file" ]; then
  ssh -O check jureca >/dev/null 2>&1 || { echo "stop: $name is not local and the JURECA link is down"; exit 4; }
  scp -q "jureca-mux:$R/$name" "$file" || { echo "stop: could not copy $name from JURECA"; exit 4; }
fi
have=$(shasum -a 256 "$file" | cut -d" " -f1)
if [ "$have" != "$want" ]; then echo "stop: sha256 of $file is $have, the queue says $want"; exit 6; fi
make check TASK=unlearning FILE="$file" 2>&1 | tail -2
until [ "$(now)" -ge "$((10#$submit_at))" ]; do sleep 2; done
date +%H:%M:%S
make submit TASK=unlearning FILE="$file" 2>&1 | grep -v -E "^(api key|server|task):"
