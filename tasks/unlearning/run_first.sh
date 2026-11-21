#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/features.py" --limit 64 --out output/kkrgp/smoke
python "$HERE/csd.py" --name smoke --limit 256 --epochs 1 --holdout 0.2 --out output/kkrgp/smoke
python "$HERE/adv.py" --limit 64 --steps 2 --out output/kkrgp/smoke
python "$HERE/features.py"
COMMON="--full 25 --hedge 8 --holdout 0.2 --epochs 8"
python "$HERE/csd.py" --name hold_all $COMMON
python "$HERE/csd.py" --name hold_last2 $COMMON --blocks 2
python "$HERE/csd.py" --name hold_early6 $COMMON --early 6
python "$HERE/csd.py" --name hold_nohead $COMMON --freeze-head
python "$HERE/adv.py" --full 25
