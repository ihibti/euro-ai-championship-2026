#!/bin/bash
set -uo pipefail
cd SCRATCHPAD/local_anchor || exit 9
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
L1=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*)
F=round1.jsonl,night-d.jsonl,night-e.jsonl
echo "B starts $(date +%H:%M:%S)"
uv run --with mlx-lm python fp8lr.py --model $M15 --label B --endpoints '^ep-(03|06|07|21|22|26|28|30|32)$' --files $F --schemes plain,tensor_merged
echo "B done $(date +%H:%M:%S)"
uv run --with mlx-lm python fp8lr.py --model $L1 --label C --date "22 Nov 2026" --endpoints '^ep-(05|09|10|11|13|15|19|27|31)$' --files $F --schemes plain,tensor_merged
echo "C done $(date +%H:%M:%S)"
uv run --with mlx-lm python fp8lr.py --model $M3 --label A --endpoints '^ep-(01|04|12|14|17|18|24|25|33)$' --files $F --schemes plain,tensor_merged
echo "A done $(date +%H:%M:%S)"
