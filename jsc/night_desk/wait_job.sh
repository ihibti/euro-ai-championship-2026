#!/bin/bash
set -uo pipefail
job="$1"
declare -a pairs=("jureca03 jureca-mux2" "jureca jureca-mux" "jureca11 jureca-mux3")
while :; do
  answered=0
  for pair in "${pairs[@]}"; do
    master=${pair% *}; alias=${pair#* }
    ssh -O check "$master" >/dev/null 2>&1 || continue
    out=$(ssh "$alias" "squeue -j $job -h -o %T 2>/dev/null; echo rc=\$?" 2>/dev/null) || continue
    case "$out" in *rc=*) answered=1; state=$(echo "$out" | grep -v '^rc=' | head -1); break;; esac
  done
  if [ "$answered" -eq 0 ]; then echo "$(date +%H:%M:%S) stop: no open connection answered"; exit 4; fi
  if [ -z "$state" ]; then echo "$(date +%H:%M:%S) job $job has left the queue (asked through $alias)"; exit 0; fi
  sleep 60
done
