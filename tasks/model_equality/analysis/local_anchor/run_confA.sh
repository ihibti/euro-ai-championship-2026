set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
until grep -q ALLDONE run_nullA2.log; do sleep 3; done
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
M7=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-8bit/snapshots/*)
SEL='^ep-(04|12|24)$'
IDS=txt_story,txt_health,idn_identity,e2_leak_quote,bat_trivia
uv run --with mlx-lm python score.py --model $M15 --label sel-qwen2.5-1.5b --rep_pen 1.05 --endpoints "$SEL" --ids $IDS --out sel_A_qwen2.5-1.5b.jsonl
uv run --with mlx-lm python score.py --model $M7 --label sel-qwen2.5-7b-q8 --rep_pen 1.05 --endpoints "$SEL" --ids $IDS --out sel_A_qwen2.5-7b-q8.jsonl
echo ALLDONE
