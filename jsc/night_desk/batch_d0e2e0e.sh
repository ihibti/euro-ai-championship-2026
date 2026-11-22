#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/d0e2e0e/tasks/unlearning
bash $REPO/run_eighth.sh "0 1 2 3"
code=$?
if [ ! -s $OUT/eighth/logits.npz ]; then
  good=""
  for n in r8_f15_rm3 r8_f20_rm3 r8_f15_e2_rm3 r8_f15_rm10 k8_e4t2_rm3 k8_e3t2_rm3_ep40 k7_e3t2_rm k7_e3t2; do
    if [ -s $OUT/$n.pt ]; then good="$good,$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python $REPO/dump_logits.py --names "${good#,}" --views exact,jitter --out $OUT/eighth/logits.npz > $OUT/eighth/log_dump.txt 2>&1 || echo "second dump failed" >> $OUT/eighth/failed.txt
  (cd $OUT && sha256sum r8_*.pt k8_*.pt > eighth/sha256.txt 2>/dev/null)
fi
exit $code
