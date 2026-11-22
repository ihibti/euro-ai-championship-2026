#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash $REPO/run_ninth.sh "0 1 2 3"
code=$?
if [ ! -s $OUT/seventh/logits.npz ]; then
  good=""
  for n in  k9_e3t2_lr2e4 k9_e3t2_lr1e4_ep40 k9_e3t2_lr2e4_ep40 k9_e3t3_lr1e4 k9_e3t1_lr1e4 k9_e35t2_lr1e4; do
    if [ -s $OUT/$n.pt ]; then good="$good,$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python $REPO/dump_logits.py --names "${good#,}" --views exact,jitter --out $OUT/seventh/logits.npz > $OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> $OUT/seventh/failed.txt
  (cd $OUT && sha256sum k9_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit $code
