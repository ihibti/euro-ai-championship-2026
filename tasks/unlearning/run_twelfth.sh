#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
LOGS=$OUT/seventh
X="--targets $OUT/xfit_extras.npz --extrapolate 3"
BASE="--full 0 --hedge 0 --blocks 6 --retain-mix 1"
export PYTHONDONTWRITEBYTECODE=1
SEEDS=(3 4 5 6 7 8 9 10)
RUNS=()
for s in "${SEEDS[@]}"; do
  RUNS+=("--name k12_b_s$s $X --tau 2 --lr 1e-4 --epochs 25 --seed $s")
done
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
MEMBERS="$OUT/k8_e3t2_rm1_lr1e4.pt $OUT/k11_b_s1.pt $OUT/k11_b_s2.pt"
for s in "${SEEDS[@]}"; do
  if [[ -f $OUT/k12_b_s$s.pt ]]; then MEMBERS="$MEMBERS $OUT/k12_b_s$s.pt"; fi
done
python "$HERE/soup.py" --models $MEMBERS --scale 0.5 --save $OUT/k12_soup_x2.pt > "$LOGS/log_soup.txt" 2>&1 || echo "soup failed" >> "$LOGS/failed.txt"
CANDS="k12_b_s3,k12_b_s4,k12_b_s5,k12_b_s6,k12_b_s7,k12_b_s8,k12_b_s9,k12_b_s10,k11_b_soup3_x2,k12_soup_x2"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "$CANDS" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k12_*.pt 2>/dev/null | tee seventh/sha256.txt
