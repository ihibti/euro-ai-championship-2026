set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
while pgrep -f 'decoder.py --model' > /dev/null; do sleep 3; done
echo DECA
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
M15=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/*)
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
L1=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*)
L3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-bf16/snapshots/*)
SEL='^ep-(10|13)$'
IDS=txt_story,txt_health,idn_identity,bat_trivia
uv run --with mlx-lm python score.py --model $L1 --label sel-llama3.2-1b --date "22 Nov 2026" --endpoints "$SEL" --ids $IDS --out sel_C_llama3.2-1b.jsonl
uv run --with mlx-lm python score.py --model $L3 --label sel-llama3.2-3b --date "22 Nov 2026" --endpoints "$SEL" --ids $IDS --out sel_C_llama3.2-3b.jsonl
W=$(python3 -I pick.py sel_C_llama3.2-1b.jsonl sel_C_llama3.2-3b.jsonl 1b 3b)
echo "CWINNER $W"
if [ "$W" = "1b" ]; then MC=$L1; else MC=$L3; fi
uv run --with mlx-lm python score.py --model $MC --label llama3.2-$W --date "22 Nov 2026" --endpoints '^ep-(05|09|10|11|13|15|16|19|27|29|31)$' --ids bat_trivia,bat_choice,idn_identity,txt_health,txt_story,e1_leak_repeat,e2_leak_quote --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_C_llama3.2-$W.jsonl
echo CDONE
uv run --with mlx-lm python stop.py --model $MS --endpoints '^ref-0[1-9]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl --tau 0.72 --out stop_refs.jsonl
echo STOPREFS
uv run --with mlx-lm python stop.py --model $M3 --endpoints '^ep-(01|04|12|14|17|18|24|25|33)$' --tau 0.70 --out stop_A.jsonl
echo STOPA
uv run --with mlx-lm python stop.py --model $M15 --endpoints '^ep-(03|06|07|21|22|26|28|30|32)$' --tau 0.70 --out stop_B.jsonl
echo STOPB
uv run --with mlx-lm python decoder.py --model $M15 --label qwen2.5-1.5b --endpoints '^ep-(07|28|21)$' --out dec_B.jsonl
echo DECB
uv run --with mlx-lm python score.py --model $M3 --label qwen2.5-3b --rep_pen 1.05 --endpoints '^ep-(01|02|04|12|14|17|18|23|24|25|33)$' --ids bat_trivia,bat_choice,idn_identity,txt_health,txt_story,e1_leak_repeat,e2_leak_quote --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_qwen2.5-3b.jsonl
echo ALLDONE
