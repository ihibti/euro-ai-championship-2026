#!/bin/bash
set -euo pipefail
GPUS="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
TARGETS=$OUT/xfit_extras.npz
TUNE="--full 0 --hedge 0 --targets $TARGETS --blocks 3 --lr 3e-5 --epochs 20"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT/fourth
read -r FIRST _ <<<"$GPUS"
CUDA_VISIBLE_DEVICES=$FIRST python "$HERE/csd.py" --name smoke_x4 --limit 300 --epochs 1 --extrapolate 2 --holdout 0.2 $TUNE --out $OUT/fourth
RUNS=(
  "--name x4_e15 --extrapolate 1.5"
  "--name x4_e20 --extrapolate 2.0"
  "--name x4_e30 --extrapolate 3.0"
  "--name x4_e45 --extrapolate 4.5"
  "--name d4_e30_probe --extrapolate 3.0 --holdout 0.2 --spare-classes 6"
)
read -r -a DEVICES <<<"$GPUS"
for slot in "${!DEVICES[@]}"; do
  (
    for i in "${!RUNS[@]}"; do
      if (( i % ${#DEVICES[@]} == slot )); then
        CUDA_VISIBLE_DEVICES=${DEVICES[$slot]} python "$HERE/csd.py" ${RUNS[$i]} $TUNE > "$OUT/fourth/log_$i.txt" 2>&1
      fi
    done
  ) &
done
wait
tail -n 12 $OUT/fourth/log_*.txt
cd $OUT && sha256sum x4_e15.pt x4_e20.pt x4_e30.pt x4_e45.pt | tee fourth/sha256.txt
