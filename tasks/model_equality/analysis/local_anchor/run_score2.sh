set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
uv run --with mlx-lm python score.py --model $M3 --label qwen2.5-3b --rep_pen 1.05 --endpoints '^(ep-(01|02|04|12|14|17|18|23|24|25|33)|local-qwen2.5-3b-bf16-.*)$' --files round1.jsonl,night-d.jsonl,night-e.jsonl --local answers.jsonl --out scores_qwen2.5-3b.jsonl
echo ADONE
uv run --with mlx-lm python score.py --model $MS --label smollm2-360m --endpoints '^ref-0[1-9]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl --out scores_smollm2-360m.jsonl
echo ALLDONE
