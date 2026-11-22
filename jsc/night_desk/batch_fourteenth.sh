#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash $REPO/run_fourteenth.sh "0 1 2 3"
code=$?
if [ ! -s $OUT/seventh/logits.npz ]; then
  good=""
  for n in k14_t15_lr1e4 k14_t15_lr8e5 k14_t1_lr9e5 k14_t1_lr75; do
    if [ -s $OUT/$n.pt ]; then good="$good,$n"; fi
  done
  if [ -n "$good" ]; then
    CUDA_VISIBLE_DEVICES=0 python $REPO/dump_logits.py --names "${good#,}" --views exact,jitter --out $OUT/seventh/logits.npz > $OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> $OUT/seventh/failed.txt
  fi
  (cd $OUT && sha256sum k14_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit $code
