#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash $REPO/run_tenth.sh "0 1 2 3"
code=$?
if [ ! -s $OUT/seventh/logits.npz ]; then
  good=""
  for n in  k10_t2_lr2e4_ep80 k10_t2_lr3e4_ep60 k10_t2_lr4e4_ep40 k10_t15_lr2e4_ep40 k10_t2_lr2e4_ep40_rm2 k10_t2_lr2e4_ep40_s1; do
    if [ -s $OUT/$n.pt ]; then good="$good,$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python $REPO/dump_logits.py --names "${good#,}" --views exact,jitter --out $OUT/seventh/logits.npz > $OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> $OUT/seventh/failed.txt
  (cd $OUT && sha256sum k10_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit $code
