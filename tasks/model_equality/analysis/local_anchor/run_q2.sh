set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
until grep -q ALLDONE run_a_leak.log; do sleep 3; done
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
M8=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-8bit/snapshots/*)
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-default --genconf genconf/Qwen__Qwen2.5-3B-Instruct.json --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 11
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-t1.0 --temp 1.0 --genconf genconf/Qwen__Qwen2.5-3B-Instruct.json --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 12
uv run --with mlx-lm python gen.py --model $M8 --label local-qwen2.5-7b-q8-default --genconf genconf/Qwen__Qwen2.5-7B-Instruct.json --probes bat_trivia,bat_choice,idn_identity,d1_repeat150 --n 10 --batch 10 --seed 13
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-default --genconf genconf/Qwen__Qwen2.5-3B-Instruct.json --probes bat_trivia,bat_choice,idn_identity,d1_repeat150 --n 10 --batch 10 --seed 14
echo ALLDONE
