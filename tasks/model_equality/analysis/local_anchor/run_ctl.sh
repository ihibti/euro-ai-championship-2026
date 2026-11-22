cd SCRATCHPAD/local_anchor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN
M3=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-bf16/snapshots/*)
MS=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/*)
F=round1.jsonl,night-d.jsonl,night-e.jsonl
cnt() { python3 -I count_fresh.py $1; }
fresh() { uv run --with mlx-lm python stop.py --model $MS --endpoints '^ref-0[12]$' --files night-b.jsonl,round1.jsonl,night-d.jsonl --tau 0.72 --out stop_refs_fresh.jsonl; }
runA() { uv run --with mlx-lm python stop.py --model $M3 --eos 151645,151643 --endpoints '^ep-(01|02|04|12|14|17|18|23|24|25|33)$' --files $F --tau 0.7 --out stop_A.jsonl & APID=$!; }
pauseA() { pkill -INT -f 'bin/python stop.py --model .*Qwen2.5-3B'; wait $APID 2>/dev/null; while pgrep -f 'stop.py --model .*Qwen2.5-3B' > /dev/null; do sleep 1; done; python3 -I trim_partial.py stop_A.jsonl; }
while pgrep -f 'stop.py --model' > /dev/null; do sleep 3; done
echo STOPB
runA
S12=0
while kill -0 $APID 2>/dev/null; do
  if [ $S12 = 0 ] && [ "$(cnt s1)" -ge 40 ] && [ "$(cnt s2)" -ge 40 ]; then
    pauseA; echo "FRESH12_ROWS s1=$(cnt s1) s2=$(cnt s2)"; fresh; echo FRESH12; S12=1; runA
  fi
  sleep 5
done
echo STOPA
if [ $S12 = 0 ]; then
  i=0; while { [ "$(cnt s1)" -lt 40 ] || [ "$(cnt s2)" -lt 40 ]; } && [ $i -lt 60 ]; do sleep 5; i=$((i+1)); done
  echo "FRESH12_ROWS s1=$(cnt s1) s2=$(cnt s2)"; fresh; echo FRESH12
fi
i=0; while [ "$(cnt s3)" -lt 160 ] && [ $i -lt 180 ]; do sleep 5; i=$((i+1)); done
echo "FRESH3_ROWS s3=$(cnt s3)"; fresh; echo FRESH3
uv run --with mlx-lm python fp8lr.py --model $MS --label refs --endpoints '^ref-0[1269]$' --files round1.jsonl,night-a.jsonl,night-b.jsonl,night-d.jsonl,night-e.jsonl,cand1.jsonl
echo FP8REFS
