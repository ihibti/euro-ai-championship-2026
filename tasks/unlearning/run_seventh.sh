#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
X="--targets $OUT/xfit_extras.npz"
STRONG="--full 0 --hedge 0 --blocks 6 --lr 5e-5 --epochs 25"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/seventh
RUNS=(
  "--name k7_e3t2_rm $X --extrapolate 3 --tau 2 --retain-mix 1"
  "--name d7_e3t2_rm_probe $X --extrapolate 3 --tau 2 --retain-mix 1 $PROBE"
  "--name d7_e3t2_probe $X --extrapolate 3 --tau 2 $PROBE"
  "--name k7_e3t2 $X --extrapolate 3 --tau 2"
  "--name k7_e3t3_rm $X --extrapolate 3 --tau 3 --retain-mix 1"
  "--name k7_e4t4_rm $X --extrapolate 4 --tau 4 --retain-mix 1"
  "--name k7_e3t2_rm3 $X --extrapolate 3 --tau 2 --retain-mix 3"
  "--name d7_e3t2_rm3_probe $X --extrapolate 3 --tau 2 --retain-mix 3 $PROBE"
  "--name u7_t32_rm --tau 32 --retain-mix 1"
  "--name d7_t32_rm_probe --tau 32 --retain-mix 1 $PROBE"
  "--name k7_e3t2_all_rm $X --extrapolate 3 --tau 2 --retain-mix 1 --blocks -1 --lr 3e-5 --batch 64"
  "--name d7_e3t2_all_rm_probe $X --extrapolate 3 --tau 2 --retain-mix 1 --blocks -1 --lr 3e-5 --batch 64 $PROBE"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $STRONG ${RUNS[$i]} > "$OUT/seventh/log_$i.txt" 2>&1 || echo "run $i failed" >> "$OUT/seventh/failed.txt"
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/seventh/log_*.txt
cd $OUT && sha256sum k7_*.pt u7_*.pt 2>/dev/null | tee seventh/sha256.txt
