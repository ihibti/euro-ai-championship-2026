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
  "--name d7_e3t2_rm_probe $X --extrapolate 3 --tau 2 --retain-mix 1 $PROBE"
  "--name d7_e3t2_probe $X --extrapolate 3 --tau 2 $PROBE"
  "--name d7_e3t2_rm3_probe $X --extrapolate 3 --tau 2 --retain-mix 3 $PROBE"
  "--name d7_e3t2_b3_rm3_probe $X --extrapolate 3 --tau 2 --retain-mix 3 --blocks 3 --lr 3e-5 --epochs 20 $PROBE"
  "--name d7_e3t2_ex_rm3_probe $X --extrapolate 3 --tau 2 --retain-mix 3 --exact-share 0.6 --aug-strength 0.5 $PROBE"
  "--name d7_e2t2_rm3_probe $X --extrapolate 2 --tau 2 --retain-mix 3 $PROBE"
  "--name d7_t32_rm3_probe --tau 32 --retain-mix 3 $PROBE"
  "--name k7_e3t2_rm3 $X --extrapolate 3 --tau 2 --retain-mix 3"
  "--name k7_e3t2_b3_rm3 $X --extrapolate 3 --tau 2 --retain-mix 3 --blocks 3 --lr 3e-5 --epochs 20"
  "--name k7_e3t2_ex_rm3 $X --extrapolate 3 --tau 2 --retain-mix 3 --exact-share 0.6 --aug-strength 0.5"
  "--name k7_e2t2_rm3 $X --extrapolate 2 --tau 2 --retain-mix 3"
  "--name u7_t32_rm3 --tau 32 --retain-mix 3"
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
CANDS="x3_ext_m80,x4_e20,x5_e25,x5_e30,t6_t3,t6_t4,t6_t5,k7_e3t2_rm3,k7_e3t2_b3_rm3,k7_e3t2_ex_rm3,k7_e2t2_rm3,u7_t32_rm3"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$OUT/seventh/logits.npz" > "$OUT/seventh/log_dump.txt" 2>&1 || echo "dump failed" >> "$OUT/seventh/failed.txt"
cd $OUT && sha256sum k7_*.pt u7_*.pt 2>/dev/null | tee seventh/sha256.txt
