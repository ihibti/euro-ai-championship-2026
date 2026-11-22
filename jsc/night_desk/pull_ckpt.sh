#!/bin/bash
set -uo pipefail
R=/p/scratch/training2666/hibti1/Begg/unlearning/output/kkrgp
D=$HOME/orca/workspaces/euro-ai-championship-2026/brill/data/unlearning/candidates
for name in "$@"; do
  ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 4; }
  if [ -f "$D/$name" ]; then echo "$name already local"; continue; fi
  want=$(ssh jureca-mux2 "sha256sum $R/$name" | cut -d" " -f1)
  scp -q "jureca-mux2:$R/$name" "$D/$name.part" || { echo "stop: copy of $name failed"; rm -f "$D/$name.part"; exit 5; }
  have=$(shasum -a 256 "$D/$name.part" | cut -d" " -f1)
  if [ "$have" != "$want" ]; then echo "stop: $name sha256 $have, JURECA says $want"; rm -f "$D/$name.part"; exit 6; fi
  mv "$D/$name.part" "$D/$name"
  echo "$(date +%H:%M:%S) $name $have"
done
