cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
while ! grep -q -E 'STOPA|Error|Traceback' run_stop.log; do sleep 5; done
while pgrep -f 'stop.py --model' > /dev/null; do sleep 3; done
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
uv run --with mlx-lm python fp8lr.py --model $MS --label refs --endpoints '^ref-0[1269]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl
echo FP8REFS
