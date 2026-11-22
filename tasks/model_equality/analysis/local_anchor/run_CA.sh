set -e
cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
until grep -q ALLDONE run_B.log; do sleep 3; done
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
uv run --with mlx-lm python decoder.py --model $M3 --label qwen2.5-3b --endpoints '^ep-(14|33|24|01|12|04)$' --out dec_A.jsonl
echo DECA
uv run --with mlx-lm python decoder.py --model $MS --label smollm2-360m --endpoints '^ref-0(1|8|4|7|9)$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl --out dec_refs.jsonl
echo DECREFS
L1=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*)
L3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-bf16/snapshots/*)
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
SEL='^ep-(10|13)$'
IDS=txt_story,txt_health,idn_identity,bat_trivia
uv run --with mlx-lm python score.py --model $L1 --label sel-llama3.2-1b --date "22 Nov 2026" --endpoints "$SEL" --ids $IDS --out sel_C_llama3.2-1b.jsonl
uv run --with mlx-lm python score.py --model $L3 --label sel-llama3.2-3b --date "22 Nov 2026" --endpoints "$SEL" --ids $IDS --out sel_C_llama3.2-3b.jsonl
W=$(python3 -I pick.py sel_C_llama3.2-1b.jsonl sel_C_llama3.2-3b.jsonl 1b 3b)
echo "CWINNER $W"
if [ "$W" = "1b" ]; then MC=$L1; else MC=$L3; fi
uv run --with mlx-lm python score.py --model $MC --label llama3.2-$W --date "22 Nov 2026" --endpoints '^ep-(05|09|10|11|13|15|16|19|27|29|31)$' --files round1.jsonl,night-d.jsonl,night-e.jsonl --out scores_C_llama3.2-$W.jsonl
echo CDONE
G=genconf/Qwen__Qwen2.5-3B-Instruct.json
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-greedy --temp 0 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes bat_trivia,bat_choice,e1_leak_repeat,e2_leak_quote,idn_identity --n 1 --batch 1 --seed 40
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-plainT0.7 --temp 0.7 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes bat_trivia,bat_choice --n 40 --batch 20 --seed 41
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-plainT0.7 --temp 0.7 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes e1_leak_repeat,e2_leak_quote,idn_identity --n 20 --batch 20 --seed 42
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-plainT1.0 --temp 1.0 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes bat_trivia,bat_choice,e1_leak_repeat,e2_leak_quote,idn_identity --n 20 --batch 20 --seed 43
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-plainT0.7 --temp 0.7 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes txt_story --n 100 --batch 25 --seed 44
uv run --with mlx-lm python gen.py --model $M3 --label local-qwen2.5-3b-bf16-plainT1.0 --temp 1.0 --top_p 1.0 --top_k 0 --rep_pen 1.0 --genconf $G --probes txt_story --n 40 --batch 20 --seed 45
echo AGENDONE
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
uv run --with mlx-lm python score.py --model $MS --label smollm2-360m --endpoints '^ref-0[1-9]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl --out scores_smollm2-360m.jsonl
echo REFS2DONE
uv run --with mlx-lm python score.py --model $M3 --label qwen2.5-3b --rep_pen 1.05 --endpoints '^(ep-(01|02|04|12|14|17|18|23|24|25|33)|local-qwen2.5-3b-bf16-plain.*)$' --files round1.jsonl,night-d.jsonl,night-e.jsonl --local answers.jsonl --out scores_qwen2.5-3b.jsonl
echo ALLDONE
