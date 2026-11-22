#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
LOGS=$OUT/seventh
X="--targets $OUT/xfit_extras.npz --extrapolate 3 --epochs 25"
BASE="--full 0 --hedge 0 --blocks 6 --retain-mix 1"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
RUNS=(
  "--name k14_t15_lr1e4 $X --tau 1.5 --lr 1e-4"
  "--name k14_t15_lr8e5 $X --tau 1.5 --lr 8e-5"
  "--name k14_t1_lr9e5 $X --tau 1 --lr 9e-5"
  "--name k14_t1_lr75 $X --tau 1 --lr 7.5e-5"
  "--name d14_t15_lr1e4_probe $X --tau 1.5 --lr 1e-4 $PROBE"
  "--name d14_t15_lr8e5_probe $X --tau 1.5 --lr 8e-5 $PROBE"
  "--name d14_t1_lr9e5_probe $X --tau 1 --lr 9e-5 $PROBE"
  "--name d14_t1_lr75_probe $X --tau 1 --lr 7.5e-5 $PROBE"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $BASE ${RUNS[$i]} > "$LOGS/log_14_$i.txt" 2>&1 || echo "run $i failed" >> "$LOGS/failed.txt"
      fi
    done
  ) &
done
wait
tail -n 12 $LOGS/log_14_*.txt
T1="$OUT/k9_e3t1_lr1e4.pt $OUT/k9_e3t1_lr1e4.pt $OUT/k9_e3t1_lr1e4.pt $OUT/k8_e3t1_rm1.pt"
T15="$OUT/k8_e3t15_rm1.pt $OUT/k10_t15_lr2e4_ep40.pt"
CANDS="k14_t15_lr1e4,k14_t15_lr8e5,k14_t1_lr9e5,k14_t1_lr75"
if ls $T1 > /dev/null 2>&1; then
  python "$HERE/soup.py" --models $T1 --save $OUT/m14_t1_blend3.pt > "$LOGS/log_14_blend_t1.txt" 2>&1 && CANDS="$CANDS,m14_t1_blend3" || echo "t1 blend failed" >> "$LOGS/failed.txt"
fi
if ls $T15 > /dev/null 2>&1; then
  python "$HERE/soup.py" --models $T15 --save $OUT/m14_t15_blend.pt > "$LOGS/log_14_blend_t15.txt" 2>&1 && CANDS="$CANDS,m14_t15_blend" || echo "t15 blend failed" >> "$LOGS/failed.txt"
fi
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k14_*.pt m14_*.pt 2>/dev/null | tee seventh/sha256.txt
