#!/bin/bash
set -euo pipefail
GPUS="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
TARGETS=$OUT/xfit_extras.npz
TUNE="--full 0 --hedge 0 --targets $TARGETS --blocks 3 --lr 3e-5 --epochs 20"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/fifth
RUNS=(
  "--name x5_e25 --extrapolate 2.5"
  "--name x5_e30 --extrapolate 3.0"
  "--name x5_e40 --extrapolate 4.0"
  "--name x5_e30_b6 --extrapolate 3.0 --blocks 6"
  "--name d5_e30_probe --extrapolate 3.0 --holdout 0.2 --spare-classes 6"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" $TUNE ${RUNS[$i]} > "$OUT/fifth/log_$i.txt" 2>&1
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/fifth/log_*.txt
cd $OUT && sha256sum x5_e25.pt x5_e30.pt x5_e40.pt x5_e30_b6.pt | tee fifth/sha256.txt
