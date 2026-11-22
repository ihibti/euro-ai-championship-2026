#!/bin/bash
set -euo pipefail
NAME="${1:?usage: tasks/unlearning/submit.sh <candidate name in output/kkrgp, e.g. bias_full30>}"
T=/p/scratch/training2666/hibti1/Begg/unlearning
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh -t jureca "cd $T/repo && ../.venv/bin/python scripts/submit.py --task unlearning --file ../output/kkrgp/$NAME.pt"
scp -q "jureca:$T/repo/tasks/unlearning/submissions/*.json" "$HERE/submissions/" || true
