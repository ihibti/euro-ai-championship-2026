set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
until grep -q ALLDONE run_score2.log; do sleep 3; done
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
G=genconf/Qwen__Qwen2.5-3B-Instruct.json
P=bat_trivia,bat_choice,idn_identity,txt_health,txt_story,d1_repeat150
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-default --genconf $G --probes $P --n 20 --batch 20 --seed 21
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-t1.0 --temp 1.0 --genconf $G --probes $P --n 20 --batch 20 --seed 22
echo GENDONE
uv run --with mlx-lm python score.py --model $M3 --label qwen2.5-3b --rep_pen 1.05 --endpoints '^(ep-(01|02|04|12|14|17|18|23|24|25|33)|local-qwen2.5-3b-bf16-.*)$' --files round1.jsonl,night-d.jsonl,night-e.jsonl --local answers.jsonl --out scores_qwen2.5-3b.jsonl
echo ALLDONE
