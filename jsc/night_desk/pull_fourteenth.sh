#!/bin/bash
set -uo pipefail
S=SCRATCHPAD
W=$HOME/worktrees/euro-ai-championship-2026/desk-unlearning
T=/p/scratch/training2666/hibti1/Begg/unlearning
ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 4; }
n=$(ssh jureca-mux2 'squeue -u $USER -h -o %i | wc -l')
[ "$n" = "0" ] || { echo "stop: a job of ours is still in the queue"; exit 3; }
ssh jureca-mux2 "sacct -j 15905724 -X -n -o JobID,State,Elapsed,Start,End,ExitCode; cd $T/output/kkrgp; echo failed.txt:; cat seventh/failed.txt; ls -l k14_* d14_* seventh/logits.npz | awk '{print \$5, \$9}'"
O=$S/fourteenth_out
rm -rf "$O"; mkdir -p "$O"
FILES="kkrgp/k14_t15_lr1e4.json kkrgp/k14_t15_lr8e5.json kkrgp/k14_t1_lr9e5.json kkrgp/k14_t1_lr75.json kkrgp/d14_t15_lr1e4_probe.json kkrgp/d14_t15_lr8e5_probe.json kkrgp/d14_t1_lr9e5_probe.json kkrgp/d14_t1_lr75_probe.json kkrgp/seventh/log_14_0.txt kkrgp/seventh/log_14_1.txt kkrgp/seventh/log_14_2.txt kkrgp/seventh/log_14_3.txt kkrgp/seventh/log_14_4.txt kkrgp/seventh/log_14_5.txt kkrgp/seventh/log_14_6.txt kkrgp/seventh/log_14_7.txt kkrgp/seventh/log_dump.txt kkrgp/seventh/logits.npz kkrgp/seventh/sha256.txt kkrgp/seventh/failed.txt desk/eighth.out"
ssh jureca-mux2 "cd $T/output && tar -cf - $FILES" | tar -xf - -C "$O" || { echo "stop: the archive stream failed"; exit 5; }
D=$W/tasks/unlearning/reports/21/fourteenth
mkdir -p "$D"
cp "$O"/kkrgp/*.json "$O"/kkrgp/seventh/* "$D"/
cp "$O"/desk/eighth.out "$D"/slurm.out
cd "$W" || exit 9
git add tasks/unlearning/reports/21/fourteenth
git add -f tasks/unlearning/reports/21/fourteenth/logits.npz
git commit -q -m "Thirteenth batch: reports, probes, logits, hashes, log (job 15905724)" && git push -q origin desk/unlearning
echo "pushed $(git log --format=%h -1) at $(date +%H:%M:%S)"
ls -l "$D" | awk 'NR>1 {print $5, $9}' | tr '\n' ';'
echo
cat "$D/sha256.txt"
