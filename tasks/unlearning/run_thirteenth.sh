#!/bin/bash
set -euo pipefail
GPUS="${1:-0 1 2 3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
LOGS=$OUT/seventh
export PYTHONDONTWRITEBYTECODE=1
read -r -a DEVICES <<<"$GPUS"
(CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/xfit.py" --name extras_s1 --reading extras --seed 1 > "$LOGS/log_xfit_s1.txt" 2>&1 || echo "xfit s1 failed" >> "$LOGS/failed.txt") &
(CUDA_VISIBLE_DEVICES=${DEVICES[1]} python "$HERE/xfit.py" --name extras_s2 --reading extras --seed 2 > "$LOGS/log_xfit_s2.txt" 2>&1 || echo "xfit s2 failed" >> "$LOGS/failed.txt") &
wait
for s in 1 2; do
  if [[ ! -f $OUT/xfit_extras_s$s.npz ]]; then echo "no xfit_extras_s$s.npz, stopping" | tee -a "$LOGS/failed.txt"; exit 1; fi
done
MIX="--full 0 --hedge 0 --blocks 6 --retain-mix 1 --targets $OUT/xfit_extras.npz --extra-targets $OUT/xfit_extras_s1.npz $OUT/xfit_extras_s2.npz --extrapolate 3 --tau 2 --lr 1e-4 --epochs 25"
(CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/csd.py" $MIX --name k13_mix_s0 --seed 0 > "$LOGS/log_mix_s0.txt" 2>&1 || echo "mix s0 failed" >> "$LOGS/failed.txt") &
(CUDA_VISIBLE_DEVICES=${DEVICES[1]} python "$HERE/csd.py" $MIX --name k13_mix_s1 --seed 1 > "$LOGS/log_mix_s1.txt" 2>&1 || echo "mix s1 failed" >> "$LOGS/failed.txt") &
(CUDA_VISIBLE_DEVICES=${DEVICES[2]} python "$HERE/csd.py" $MIX --name d13_mix_probe --seed 0 --holdout 0.2 --spare-classes 6 > "$LOGS/log_mix_probe.txt" 2>&1 || echo "mix probe failed" >> "$LOGS/failed.txt") &
wait
tail -n 12 $LOGS/log_mix_*.txt
python "$HERE/soup.py" --models $OUT/k13_mix_s0.pt $OUT/k13_mix_s1.pt --scale 0.5 --save $OUT/k13_mix_soup_x2.pt > "$LOGS/log_soup.txt" 2>&1 || echo "soup failed" >> "$LOGS/failed.txt"
CUDA_VISIBLE_DEVICES=${DEVICES[0]} python "$HERE/dump_logits.py" --names "k13_mix_s0,k13_mix_s1,k13_mix_soup_x2" --views exact,jitter --out "$LOGS/logits.npz" > "$LOGS/log_dump.txt" 2>&1 || echo "dump failed" >> "$LOGS/failed.txt"
cd $OUT && sha256sum k13_*.pt 2>/dev/null | tee seventh/sha256.txt
