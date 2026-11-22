#!/bin/bash
limit="${1:-60}"
idle=0
while [ "$idle" -lt "$limit" ]; do
  sleep 60
  if [ -e output/desk/stop ]; then break; fi
  n=$(squeue -h -s -j "$SLURM_JOB_ID" -o %i 2>/dev/null | grep -c -v -E '\.(0|batch|extern)$')
  n="${n:-0}"
  if [ "$n" -gt 0 ]; then idle=0; else idle=$((idle + 1)); fi
done
echo "holder ends after $idle idle minutes at $(date +%H:%M)"
