#!/bin/bash
set -uo pipefail
cd SCRATCHPAD/local_anchor || exit 9
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
while pgrep -f 'run_fp8_families.sh' > /dev/null; do sleep 5; done
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
echo "stop A pair starts $(date +%H:%M:%S)"
uv run --with mlx-lm python stop.py --model $M3 --endpoints '^ep-(12|18)$' --tau 0.70 --out stop_A.jsonl
echo "stop A pair done $(date +%H:%M:%S)"
