set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M05=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/*)
uv run --with mlx-lm python score.py --model $M05 --label qwen2.5-0.5b --rep_pen 1.1 --endpoints '^ep-(03|06|07|08|20|21|22|26|28|30|32)$' --ids bat_trivia,bat_choice,idn_identity,txt_health,txt_story,e1_leak_repeat,e2_leak_quote --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_B_qwen2.5-0.5b.jsonl
echo B05DONE
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
IDS=bat_trivia,idn_identity,txt_health,txt_story
uv run --with mlx-lm python score.py --model $M15 --label sel-qwen2.5-1.5b --rep_pen 1.1 --endpoints '^ep-(07|21)$' --ids $IDS --out sel_B_qwen2.5-1.5b.jsonl
python3 -I -c "
import json
a=[json.loads(l) for l in open('scores_B_qwen2.5-0.5b.jsonl') if l.strip()]
ids=set('$IDS'.split(','))
open('sel_B_qwen2.5-0.5b.jsonl','w').write(''.join(json.dumps(r)+chr(10) for r in a if r['endpoint'] in ('ep-07','ep-21') and r['id'] in ids))
"
W=$(python3 -I pick.py sel_B_qwen2.5-0.5b.jsonl sel_B_qwen2.5-1.5b.jsonl 0.5b 1.5b)
echo "BWINNER $W"
if [ "$W" = "1.5b" ]; then
uv run --with mlx-lm python score.py --model $M15 --label qwen2.5-1.5b --rep_pen 1.1 --endpoints '^ep-(03|06|07|08|20|21|22|26|28|30|32)$' --ids bat_trivia,bat_choice,idn_identity,txt_health,txt_story,e1_leak_repeat,e2_leak_quote --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_B_qwen2.5-1.5b.jsonl
fi
echo B15DONE
uv run --with mlx-lm python score.py --model $M15 --label sel-qwen2.5-1.5b --rep_pen 1.05 --endpoints '^ep-(12|24)$' --ids txt_story,txt_health,idn_identity,bat_trivia --out sel_A_qwen2.5-1.5b.jsonl
echo ALLDONE
