#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="--full 30 --hedge 16 --holdout 0.2 --spare-classes 6 --epochs 20"
python "$HERE/csd.py" --name s2_last2_lr3 $COMMON --blocks 2 --lr 3e-5
python "$HERE/csd.py" --name s2_last4_lr3 $COMMON --blocks 4 --lr 3e-5
python "$HERE/csd.py" --name s2_all_lr1 $COMMON --lr 1e-5
python "$HERE/csd.py" --name s2_last2_lr10 $COMMON --blocks 2 --lr 1e-4
