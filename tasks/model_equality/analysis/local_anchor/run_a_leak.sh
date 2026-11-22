set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M8=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-8bit/snapshots/*)
M4=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/*)
G=genconf/Qwen__Qwen2.5-7B-Instruct.json
uv run --with mlx-lm python gen.py --model $M8 --label local-qwen2.5-7b-q8-default --genconf $G --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 1
uv run --with mlx-lm python gen.py --model $M8 --label local-qwen2.5-7b-q8-t1.0 --temp 1.0 --genconf $G --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 2
uv run --with mlx-lm python gen.py --model $M4 --label local-qwen2.5-7b-q4-default --genconf $G --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 3
uv run --with mlx-lm python gen.py --model $M4 --label local-qwen2.5-7b-q4-t1.0 --temp 1.0 --genconf $G --probes e1_leak_repeat,e2_leak_quote --n 40 --batch 20 --seed 4
echo ALLDONE
