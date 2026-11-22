import json
import sys

A = "$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers"
sys.path.insert(0, "SCRATCHPAD/local_anchor")
from stopfresh import stage_of

want = sys.argv[1]
n = 0
for f in ("night-b.jsonl", "round1.jsonl", "night-d.jsonl"):
    for line in open(f"{A}/{f}"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["endpoint"] in ("ref-01", "ref-02") and stage_of(f"{f}|{r['endpoint']}|{r['id']}|{r['repeat']}") == want:
            n += 1
print(n)
