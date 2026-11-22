#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash $REPO/run_eighth.sh "0 1 2 3"
code=$?
if [ ! -s $OUT/eighth/logits.npz ]; then
  good=""
  for n in k8_e4t2_rm1 k8_e3t15_rm1 k8_e3t1_rm1 k8_e4t2_rm3 k8_e3t2_rm1_ep40 k8_e3t2_rm1_lr1e4 k8_e5t25_rm1 k8_e3t2_rm1_b9 k7_e3t2_rm k7_e3t2; do
    if [ -s $OUT/$n.pt ]; then good="$good,$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python $REPO/dump_logits.py --names "${good#,}" --views exact,jitter --out $OUT/eighth/logits.npz > $OUT/eighth/log_dump.txt 2>&1 || echo "second dump failed" >> $OUT/eighth/failed.txt
  (cd $OUT && sha256sum k8_*.pt > eighth/sha256.txt 2>/dev/null)
fi
exit $code
