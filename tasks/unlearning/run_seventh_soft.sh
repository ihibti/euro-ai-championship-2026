#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
X="--targets $OUT/xfit_extras.npz"
STRONG="--full 0 --hedge 0 --blocks 6 --lr 5e-5 --epochs 25"
PROBE="--holdout 0.2 --spare-classes 6"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/seventh_soft
GENTLE="--blocks 3 --lr 3e-5 --epochs 20"
RUNS=(
  "--name d7s_t4_probe --tau 4 $GENTLE $PROBE"
  "--name d7s_t4_rm3_probe --tau 4 --retain-mix 3 $GENTLE $PROBE"
  "--name d7s_t6_rm3_probe --tau 6 --retain-mix 3 $GENTLE $PROBE"
  "--name k7s_t4_rm3 --tau 4 --retain-mix 3 $GENTLE"
  "--name k7s_t5_rm3 --tau 5 --retain-mix 3 $GENTLE"
  "--name k7s_t6_rm3 --tau 6 --retain-mix 3 $GENTLE"
  "--name k7s_t8_rm3 --tau 8 --retain-mix 3 $GENTLE"
  "--name k7s_x05t4_rm3 $X --extrapolate 0.5 --tau 4 --retain-mix 3 $GENTLE"
  "--name d7s_x05t4_rm3_probe $X --extrapolate 0.5 --tau 4 --retain-mix 3 $GENTLE $PROBE"
  "--name k7_e3t2_rm3 $X --extrapolate 3 --tau 2 --retain-mix 3"
  "--name d7_e3t2_rm3_probe $X --extrapolate 3 --tau 2 --retain-mix 3 $PROBE"
  "--name u7_t32_rm3 --tau 32 --retain-mix 3"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $STRONG ${RUNS[$i]} > "$OUT/seventh_soft/log_$i.txt" 2>&1 || echo "run $i failed" >> "$OUT/seventh_soft/failed.txt"
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/seventh_soft/log_*.txt
CANDS="x3_ext_m80,x4_e20,t6_t3,t6_t4,t6_t5,k7s_t4_rm3,k7s_t5_rm3,k7s_t6_rm3,k7s_t8_rm3,k7s_x05t4_rm3,k7_e3t2_rm3,u7_t32_rm3"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$OUT/seventh_soft/logits.npz" > "$OUT/seventh_soft/log_dump.txt" 2>&1 || echo "dump failed" >> "$OUT/seventh_soft/failed.txt"
cd $OUT && sha256sum k7*.pt u7_*.pt 2>/dev/null | tee seventh_soft/sha256.txt
