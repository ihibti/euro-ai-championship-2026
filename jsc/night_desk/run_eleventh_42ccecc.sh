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
  "--name k11_b_s1 $X --tau 2 --lr 1e-4 --epochs 25 --seed 1"
  "--name d11_b_s1_probe $X --tau 2 --lr 1e-4 --epochs 25 --seed 1 $PROBE"
  "--name k11_b_ep20 $X --tau 2 --lr 1e-4 --epochs 20"
  "--name d11_b_ep20_probe $X --tau 2 --lr 1e-4 --epochs 20 $PROBE"
  "--name k11_b_rm2 $X --tau 2 --lr 1e-4 --epochs 25 --retain-mix 2"
  "--name d11_b_rm2_probe $X --tau 2 --lr 1e-4 --epochs 25 --retain-mix 2 $PROBE"
  "--name k11_b_bl4 $X --tau 2 --lr 1.5e-4 --epochs 25 --blocks 4"
  "--name d11_b_bl4_probe $X --tau 2 --lr 1.5e-4 --epochs 25 --blocks 4 $PROBE"
  "--name k11_b_k25 --targets $OUT/xfit_extras.npz --extrapolate 2.5 --tau 2 --lr 1.2e-4 --epochs 25"
  "--name d11_b_k25_probe --targets $OUT/xfit_extras.npz --extrapolate 2.5 --tau 2 --lr 1.2e-4 --epochs 25 $PROBE"
  "--name k11_b_s2 $X --tau 2 --lr 1e-4 --epochs 25 --seed 2"
  "--name d11_b_s2_probe $X --tau 2 --lr 1e-4 --epochs 25 --seed 2 $PROBE"
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
CANDS="k11_b_s1,k11_b_ep20,k11_b_rm2,k11_b_bl4,k11_b_k25,k11_b_s2"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k11_*.pt 2>/dev/null | tee seventh/sha256.txt
