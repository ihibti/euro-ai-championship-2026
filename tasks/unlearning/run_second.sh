#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="--holdout 0.2 --spare-classes 6 --epochs 20"
python "$HERE/csd.py" --name s2_rem_last2 --full 30 --hedge 16 $COMMON --blocks 2 --lr 3e-5
python "$HERE/csd.py" --name s2_rem_all --full 30 --hedge 16 $COMMON --lr 1e-5
python "$HERE/csd.py" --name s2_ext_last2 --full 0 --hedge 0 --even 10 $COMMON --blocks 2 --lr 3e-5
python "$HERE/csd.py" --name s2_ext_all --full 0 --hedge 0 --even 10 $COMMON --lr 1e-5
python "$HERE/adv.py" --full 0 --out output/kkrgp/adv_orig
