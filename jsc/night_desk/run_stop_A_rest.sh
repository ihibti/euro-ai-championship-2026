#!/bin/bash
set -uo pipefail
cd SCRATCHPAD/local_anchor || exit 9
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
echo "stop A rest starts $(date +%H:%M:%S)"
uv run --with mlx-lm python stop.py --model $M3 --endpoints '^ep-(01|17|24|25)$' --tau 0.70 --out stop_A.jsonl
uv run --with mlx-lm python stop.py --model $M3 --endpoints '^ep-(04|14|33)$' --tau 0.70 --out stop_A.jsonl
echo "stop A rest done $(date +%H:%M:%S)"
