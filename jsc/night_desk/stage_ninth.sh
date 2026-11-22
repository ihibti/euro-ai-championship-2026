#!/bin/bash
set -euo pipefail
C=$1
S=SCRATCHPAD
B=$HOME/orca/workspaces/euro-ai-championship-2026/brill
T=/p/scratch/training2666/hibti1/Begg/unlearning
cd "$B"
git fetch -q origin
for f in csd.py dump_logits.py lib.py; do
  a=$(git show "5ac28b2:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  b=$(git show "${C}:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  [ "$a" = "$b" ] || { echo "stop: ${f} differs between 5ac28b2 and ${C}; stage the changed file too"; exit 3; }
done
git show "${C}:tasks/unlearning/run_ninth.sh" > "$S/run_ninth_${C}.sh"
bash -n "$S/run_ninth_${C}.sh"
PAIRS="k7_e3t2:k9_e3t2_lr2e4 k7_e3t2_rm3:k9_e3t2_lr1e4_ep40 k7_e3t2_b3_rm3:k9_e3t2_lr2e4_ep40 k7_e3t2_ex_rm3:k9_e3t3_lr1e4 k7_e2t2_rm3:k9_e3t1_lr1e4 u7_t32_rm3:k9_e35t2_lr1e4"
PROBES="d7_e3t2_probe:d9_e3t2_lr2e4_probe d7_e3t2_rm3_probe:d9_e3t2_lr1e4_ep40_probe d7_e3t2_b3_rm3_probe:d9_e3t2_lr2e4_ep40_probe d7_e3t2_ex_rm3_probe:d9_e3t3_lr1e4_probe d7_e2t2_rm3_probe:d9_e3t1_lr1e4_probe d7_t32_rm3_probe:d9_e35t2_lr1e4_probe"
NAMES=""
for p in $PAIRS; do NAMES="$NAMES ${p#*:}"; done
for p in $PAIRS; do
  grep -q -- "--name ${p#*:} " "$S/run_ninth_${C}.sh" || { echo "stop: ${p#*:} is not a run of ${C}"; exit 3; }
done
[ "$(grep -c -E '^ +"--name ' "$S/run_ninth_${C}.sh")" = "12" ] || { echo "stop: ${C} does not have 12 runs"; exit 3; }
LOCAL=""
for p in $PAIRS; do
  LOCAL="$LOCAL ${p%%:*}=$(shasum -a 256 "$B/data/unlearning/candidates/${p%%:*}.pt" | cut -c1-64)"
done
cat > "$S/batch_ninth.sh" <<WRAP
#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash \$REPO/run_ninth.sh "0 1 2 3"
code=\$?
if [ ! -s \$OUT/seventh/logits.npz ]; then
  good=""
  for n in $NAMES; do
    if [ -s \$OUT/\$n.pt ]; then good="\$good,\$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python \$REPO/dump_logits.py --names "\${good#,}" --views exact,jitter --out \$OUT/seventh/logits.npz > \$OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> \$OUT/seventh/failed.txt
  (cd \$OUT && sha256sum k9_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit \$code
WRAP
bash -n "$S/batch_ninth.sh"
uv run python scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=begg-ul9 TIME=00:50:00 TASK_DIR=$T "COMMAND=bash output/desk/batch_ninth.sh" > "$S/batch_ninth.sbatch"
cat > "$S/stage_ninth_remote.sh" <<REMOTE
set -euo pipefail
cd $T
[ "\$(squeue -u \$USER -h -o %i | wc -l)" = "0" ] || { echo "stop: a job of ours is in the queue"; exit 4; }
before=\$(find output runs -xdev | wc -l)
cd output/kkrgp
for kv in $LOCAL; do
  name=\${kv%%=*}; want=\${kv#*=}
  if [ -e \$name.pt ]; then
    have=\$(sha256sum \$name.pt | cut -c1-64)
    [ "\$have" = "\$want" ] || { echo "stop: \$name.pt differs from the laptop copy"; exit 5; }
  fi
done
for p in $PAIRS; do
  old=\${p%%:*}; new=\${p#*:}
  for ext in pt json; do
    if [ -e \$old.\$ext ]; then mv \$old.\$ext \$new.\$ext; fi
    [ -e \$new.\$ext ] || { echo "stop: \$new.\$ext is missing"; exit 6; }
    : > \$new.\$ext
  done
done
for p in $PROBES; do
  old=\${p%%:*}; new=\${p#*:}
  if [ -e \$old.json ]; then mv \$old.json \$new.json; fi
  [ -e \$new.json ] || { echo "stop: \$new.json is missing"; exit 6; }
  : > \$new.json
done
if [ ! -e seventh/failed.txt ]; then mv eighth/log_15.txt seventh/failed.txt; fi
for f in seventh/log_0.txt seventh/log_1.txt seventh/log_2.txt seventh/log_3.txt seventh/log_4.txt seventh/log_5.txt seventh/log_6.txt seventh/log_7.txt seventh/log_8.txt seventh/log_9.txt seventh/log_10.txt seventh/log_11.txt seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
  : > \$f
done
cd $T
R=runs/5ac28b2/tasks/unlearning
if [ -e \$R/run_eighth.sh ]; then mv \$R/run_eighth.sh \$R/run_ninth.sh; fi
if [ -e output/desk/batch_5ac28b2.sh ]; then mv output/desk/batch_5ac28b2.sh output/desk/batch_ninth.sh; fi
if [ -e output/desk/batch_5ac28b2.sbatch ]; then mv output/desk/batch_5ac28b2.sbatch output/desk/batch_ninth.sbatch; fi
: > output/desk/eighth.out
after=\$(find output runs -xdev | wc -l)
echo "files before \$before, after \$after"
test -e output/kkrgp/k7_e3t2_rm.pt && test -s output/kkrgp/k7_e3t2_rm.pt && echo "k7_e3t2_rm.pt untouched"
test -s output/kkrgp/xfit_extras.npz && echo "targets file present"
REMOTE
ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 8; }
ssh jureca-mux2 "ls -d $T >/dev/null" || { echo "stop: no scratch on this node"; exit 8; }
ssh jureca-mux2 'bash -s' < "$S/stage_ninth_remote.sh"
ssh jureca-mux2 "cat > $T/runs/5ac28b2/tasks/unlearning/run_ninth.sh" < "$S/run_ninth_${C}.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_ninth.sh" < "$S/batch_ninth.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_ninth.sbatch" < "$S/batch_ninth.sbatch"
for f in csd.py dump_logits.py lib.py run_ninth.sh; do
  a=$(git show "${C}:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/runs/5ac28b2/tasks/unlearning/${f}" | cut -c1-64)
  [ "$a" = "$b" ] && echo "${f} matches ${C}" || { echo "stop: ${f} on scratch differs from ${C}"; exit 9; }
done
ssh jureca-mux2 "cd $T && grep -c -E 'time=00:50:00|batch_ninth.sh|reservation=cispahack|gres=gpu:4|job-name=begg-ul9' output/desk/batch_ninth.sbatch && grep -c 'run_ninth.sh' output/desk/batch_ninth.sh && ls output/kkrgp/k9_* output/kkrgp/d9_* | wc -l && find output/kkrgp -maxdepth 1 -name 'k9_*' -size +0 | wc -l && df --output=iavail /p/scratch/training2666 | tail -1"
echo "staged ${C} at $(date +%H:%M:%S)"
