cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
A=$HOME/worktrees/euro-ai-championship-2026/model-equality-refs/tasks/model_equality/answers
while ! grep -q -E 'STOPA|Error|Traceback' run_stop.log; do sleep 5; done
while pgrep -f 'stop.py --model' > /dev/null; do sleep 3; done
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
dcount() { python3 -I -c "
import json
n=0
for l in open('$A/night-d.jsonl'):
    try: r=json.loads(l)
    except Exception: continue
    if r['endpoint'] in ('ref-01','ref-02') and r['id'] in ('d1_repeat150','d2_story100','d3_openlist','d4_copycont') and r['repeat']>=10: n+=1
print(n)"; }
i=0
while [ "$(dcount)" -lt 160 ] && [ $i -lt 120 ]; do sleep 5; i=$((i+1)); done
echo "FRESH_D_ROWS $(dcount)"
uv run --with mlx-lm python stop.py --model $MS --endpoints '^ref-0[12]$' --files night-d.jsonl --tau 0.72 --out stop_refs_fresh.jsonl
echo FRESHD
uv run --with mlx-lm python fp8lr.py --model $MS --label refs --endpoints '^ref-0[1269]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl
echo FP8REFS
