set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
while pgrep -f 'score.py --model' > /dev/null; do sleep 3; done
echo REFSDONE
M05=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/*)
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
B='^ep-(03|06|07|08|20|21|22|26|28|30|32)$'
uv run --with mlx-lm python score.py --model $M05 --label qwen2.5-0.5b --rep_pen 1.1 --endpoints "$B" --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_B_qwen2.5-0.5b.jsonl
echo B05DONE
uv run --with mlx-lm python score.py --model $M15 --label qwen2.5-1.5b --rep_pen 1.1 --endpoints "$B" --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_B_qwen2.5-1.5b.jsonl
echo B15DONE
uv run --with mlx-lm python score.py --model $M15 --label sel-qwen2.5-1.5b --rep_pen 1.05 --endpoints '^ep-(12|24)$' --ids txt_story,txt_health,idn_identity,bat_trivia,e2_leak_quote --out sel_A_qwen2.5-1.5b.jsonl
echo ALLDONE
