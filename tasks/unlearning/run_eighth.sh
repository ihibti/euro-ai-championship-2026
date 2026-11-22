#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
X="--targets $OUT/xfit_extras.npz"
STRONG="--full 0 --hedge 0 --blocks 6 --lr 5e-5 --epochs 25"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/eighth
RUNS=(
  "--name k8_e4t2_rm1 $X --extrapolate 4 --tau 2 --retain-mix 1"
  "--name d8_e4t2_rm1_probe $X --extrapolate 4 --tau 2 --retain-mix 1 $PROBE"
  "--name k8_e3t15_rm1 $X --extrapolate 3 --tau 1.5 --retain-mix 1"
  "--name d8_e3t15_rm1_probe $X --extrapolate 3 --tau 1.5 --retain-mix 1 $PROBE"
  "--name k8_e3t1_rm1 $X --extrapolate 3 --tau 1 --retain-mix 1"
  "--name d8_e3t1_rm1_probe $X --extrapolate 3 --tau 1 --retain-mix 1 $PROBE"
  "--name k8_e4t2_rm3 $X --extrapolate 4 --tau 2 --retain-mix 3"
  "--name d8_e4t2_rm3_probe $X --extrapolate 4 --tau 2 --retain-mix 3 $PROBE"
  "--name k8_e3t2_rm1_ep40 $X --extrapolate 3 --tau 2 --retain-mix 1 --epochs 40"
  "--name d8_e3t2_rm1_ep40_probe $X --extrapolate 3 --tau 2 --retain-mix 1 --epochs 40 $PROBE"
  "--name k8_e3t2_rm1_lr1e4 $X --extrapolate 3 --tau 2 --retain-mix 1 --lr 1e-4"
  "--name d8_e3t2_rm1_lr1e4_probe $X --extrapolate 3 --tau 2 --retain-mix 1 --lr 1e-4 $PROBE"
  "--name k8_e5t25_rm1 $X --extrapolate 5 --tau 2.5 --retain-mix 1"
  "--name d8_e5t25_rm1_probe $X --extrapolate 5 --tau 2.5 --retain-mix 1 $PROBE"
  "--name k8_e3t2_rm1_b9 $X --extrapolate 3 --tau 2 --retain-mix 1 --blocks 9"
  "--name d8_e3t2_rm1_b9_probe $X --extrapolate 3 --tau 2 --retain-mix 1 --blocks 9 $PROBE"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $STRONG ${RUNS[$i]} > "$OUT/eighth/log_$i.txt" 2>&1 || echo "run $i failed" >> "$OUT/eighth/failed.txt"
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/eighth/log_*.txt
CANDS="k8_e4t2_rm1,k8_e3t15_rm1,k8_e3t1_rm1,k8_e4t2_rm3,k8_e3t2_rm1_ep40,k8_e3t2_rm1_lr1e4,k8_e5t25_rm1,k8_e3t2_rm1_b9,k7_e3t2_rm,k7_e3t2"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$OUT/eighth/logits.npz" > "$OUT/eighth/log_dump.txt" 2>&1 || echo "dump failed" >> "$OUT/eighth/failed.txt"
cd $OUT && sha256sum k8_*.pt 2>/dev/null | tee eighth/sha256.txt
