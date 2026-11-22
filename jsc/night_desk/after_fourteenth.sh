#!/bin/bash
set -uo pipefail
S=SCRATCHPAD
bash $S/wait_job.sh 15905724 || exit $?
bash $S/pull_fourteenth.sh || exit $?
bash $S/pull_ckpt.sh k14_t15_lr1e4.pt k14_t15_lr8e5.pt k14_t1_lr9e5.pt k14_t1_lr75.pt
