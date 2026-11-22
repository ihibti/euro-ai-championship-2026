#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
LOGS=$OUT/seventh
X="--targets $OUT/xfit_extras.npz"
BASE="--full 0 --hedge 0 --blocks 6 --epochs 25 --retain-mix 1"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
RUNS=(
  "--name k9_e3t1_lr1e4 $X --extrapolate 3 --tau 1 --lr 1e-4"
  "--name d9_e3t1_lr1e4_probe $X --extrapolate 3 --tau 1 --lr 1e-4 $PROBE"
  "--name k9_e3t1_lr2e4 $X --extrapolate 3 --tau 1 --lr 2e-4"
  "--name d9_e3t1_lr2e4_probe $X --extrapolate 3 --tau 1 --lr 2e-4 $PROBE"
  "--name k9_e3t1_lr1e4_ep40 $X --extrapolate 3 --tau 1 --lr 1e-4 --epochs 40"
  "--name d9_e3t1_lr1e4_ep40_probe $X --extrapolate 3 --tau 1 --lr 1e-4 --epochs 40 $PROBE"
  "--name k9_e35t15_lr1e4 $X --extrapolate 3.5 --tau 1.5 --lr 1e-4"
  "--name d9_e35t15_lr1e4_probe $X --extrapolate 3.5 --tau 1.5 --lr 1e-4 $PROBE"
  "--name k9_e3t1_lr1e4_b9 $X --extrapolate 3 --tau 1 --lr 1e-4 --blocks 9"
  "--name d9_e3t1_lr1e4_b9_probe $X --extrapolate 3 --tau 1 --lr 1e-4 --blocks 9 $PROBE"
  "--name k9_e25t1_lr1e4 $X --extrapolate 2.5 --tau 1 --lr 1e-4"
  "--name d9_e25t1_lr1e4_probe $X --extrapolate 2.5 --tau 1 --lr 1e-4 $PROBE"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $BASE ${RUNS[$i]} > "$LOGS/log_$i.txt" 2>&1 || echo "run $i failed" >> "$LOGS/failed.txt"
      fi
    done
  ) &
done
wait
tail -n 12 $LOGS/log_*.txt
CANDS="k9_e3t1_lr1e4,k9_e3t1_lr2e4,k9_e3t1_lr1e4_ep40,k9_e35t15_lr1e4,k9_e3t1_lr1e4_b9,k9_e25t1_lr1e4"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k9_*.pt 2>/dev/null | tee seventh/sha256.txt
