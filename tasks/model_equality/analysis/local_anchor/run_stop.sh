set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
L1=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*)
F=round1.jsonl,night-d.jsonl,night-e.jsonl
uv run --with mlx-lm python stop.py --model $L1 --date "22 Nov 2026" --eos 128001,128008,128009 --endpoints '^ep-(05|09|10|11|13|15|16|19|27|29|31)$' --files $F --tau 0.7 --out stop_C.jsonl
echo STOPC
uv run --with mlx-lm python stop.py --model $M15 --eos 151645,151643 --endpoints '^ep-(03|06|07|08|20|21|22|26|28|30|32)$' --files $F --tau 0.7 --out stop_B.jsonl
echo STOPB
uv run --with mlx-lm python stop.py --model $M3 --eos 151645,151643 --endpoints '^ep-(01|02|04|12|14|17|18|23|24|25|33)$' --files $F --tau 0.7 --out stop_A.jsonl
echo STOPA
