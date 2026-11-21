#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=output/kkrgp
python "$HERE/xfit.py" --name smoke --reading extras --limit 300 --folds 2 --epochs 1 --out $OUT/smoke
python "$HERE/csd.py" --name smoke_mix --limit 300 --epochs 1 --full 0 --hedge 0 --holdout 0.2 --targets $OUT/smoke/xfit_smoke.npz --out $OUT/smoke
python "$HERE/xfit.py" --name extras --reading extras
python "$HERE/xfit.py" --name removal --reading removal
TUNE="--blocks 3 --lr 3e-5 --epochs 20"
python "$HERE/csd.py" --name d3_ext_probe --full 0 --hedge 0 --targets $OUT/xfit_extras.npz --mix 0.6 --holdout 0.2 --spare-classes 6 $TUNE
python "$HERE/csd.py" --name d3_rem_probe --full 30 --hedge 0 --targets $OUT/xfit_removal.npz --mix 0.6 --holdout 0.2 --spare-classes 6 $TUNE
python "$HERE/csd.py" --name x3_ext_m40 --full 0 --hedge 0 --targets $OUT/xfit_extras.npz --mix 0.4 $TUNE
python "$HERE/csd.py" --name x3_ext_m80 --full 0 --hedge 0 --targets $OUT/xfit_extras.npz --mix 0.8 $TUNE
python "$HERE/csd.py" --name x3_rem_m40 --full 30 --hedge 0 --targets $OUT/xfit_removal.npz --mix 0.4 $TUNE
python "$HERE/csd.py" --name x3_rem_m80 --full 30 --hedge 0 --targets $OUT/xfit_removal.npz --mix 0.8 $TUNE
python "$HERE/adv.py" --full 0 --out $OUT/adv_orig
