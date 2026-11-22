#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
LOGS=$OUT/seventh
X="--targets $OUT/xfit_extras.npz --extrapolate 3"
BASE="--full 0 --hedge 0 --blocks 6 --retain-mix 1"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
RUNS=(
  "--name k10_t2_lr2e4_ep80 $X --tau 2 --lr 2e-4 --epochs 80"
  "--name d10_t2_lr2e4_ep80_probe $X --tau 2 --lr 2e-4 --epochs 80 $PROBE"
  "--name k10_t2_lr3e4_ep60 $X --tau 2 --lr 3e-4 --epochs 60"
  "--name d10_t2_lr3e4_ep60_probe $X --tau 2 --lr 3e-4 --epochs 60 $PROBE"
  "--name k10_t2_lr4e4_ep40 $X --tau 2 --lr 4e-4 --epochs 40"
  "--name d10_t2_lr4e4_ep40_probe $X --tau 2 --lr 4e-4 --epochs 40 $PROBE"
  "--name k10_t15_lr2e4_ep40 $X --tau 1.5 --lr 2e-4 --epochs 40"
  "--name d10_t15_lr2e4_ep40_probe $X --tau 1.5 --lr 2e-4 --epochs 40 $PROBE"
  "--name k10_t2_lr2e4_ep40_rm2 $X --tau 2 --lr 2e-4 --epochs 40 --retain-mix 2"
  "--name d10_t2_lr2e4_ep40_rm2_probe $X --tau 2 --lr 2e-4 --epochs 40 --retain-mix 2 $PROBE"
  "--name k10_t2_lr2e4_ep40_s1 $X --tau 2 --lr 2e-4 --epochs 40 --seed 1"
  "--name d10_t2_lr2e4_ep40_s1_probe $X --tau 2 --lr 2e-4 --epochs 40 --seed 1 $PROBE"
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
CANDS="k10_t2_lr2e4_ep80,k10_t2_lr3e4_ep60,k10_t2_lr4e4_ep40,k10_t15_lr2e4_ep40,k10_t2_lr2e4_ep40_rm2,k10_t2_lr2e4_ep40_s1"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k10_*.pt 2>/dev/null | tee seventh/sha256.txt
