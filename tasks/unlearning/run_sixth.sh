#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
TARGETS=$OUT/xfit_extras.npz
TUNE="--full 0 --hedge 0 --blocks 3 --lr 3e-5 --epochs 20"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/sixth
for t in 2 3 4; do
  [ -f "$OUT/scale_t$t.pt" ] || python "$HERE/splice_head.py" --scale $t --save "$OUT/scale_t$t.pt"
done
RUNS=(
  "--name t6_t3 --tau 3"
  "--name t6_t4 --tau 4"
  "--name t6_t2 --tau 2"
  "--name t6_t5 --tau 5"
  "--name d6_t3_probe --tau 3 --holdout 0.2 --spare-classes 6"
  "--name d6_t4_probe --tau 4 --holdout 0.2 --spare-classes 6"
  "--name x6_k05_t3 --tau 3 --targets $TARGETS --extrapolate 0.5"
  "--name t6_t3_b6 --tau 3 --blocks 6"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $TUNE ${RUNS[$i]} > "$OUT/sixth/log_$i.txt" 2>&1
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/sixth/log_*.txt
cd $OUT && sha256sum scale_t2.pt scale_t3.pt scale_t4.pt t6_t3.pt t6_t4.pt t6_t2.pt t6_t5.pt x6_k05_t3.pt t6_t3_b6.pt | tee sixth/sha256.txt
