#!/bin/bash
set -euo pipefail
C=$1
S=SCRATCHPAD
B=$HOME/orca/workspaces/euro-ai-championship-2026/brill
T=/p/scratch/training2666/hibti1/Begg/unlearning
cd "$B"
git fetch -q origin
for f in csd.py dump_logits.py lib.py; do
  a=$(git show "66c6b31:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  b=$(git show "${C}:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  [ "$a" = "$b" ] || { echo "stop: ${f} differs between 66c6b31 and ${C}; stage the changed file too"; exit 3; }
done
git show "${C}:tasks/unlearning/run_eleventh.sh" > "$S/run_eleventh_${C}.sh"
bash -n "$S/run_eleventh_${C}.sh"
PAIRS="k10_t2_lr2e4_ep80:k11_b_s1 k10_t2_lr3e4_ep60:k11_b_ep20 k10_t2_lr4e4_ep40:k11_b_rm2 k10_t15_lr2e4_ep40:k11_b_bl4 k10_t2_lr2e4_ep40_rm2:k11_b_k25 k10_t2_lr2e4_ep40_s1:k11_b_s2"
PROBES="d10_t2_lr2e4_ep80_probe:d11_b_s1_probe d10_t2_lr3e4_ep60_probe:d11_b_ep20_probe d10_t2_lr4e4_ep40_probe:d11_b_rm2_probe d10_t15_lr2e4_ep40_probe:d11_b_bl4_probe d10_t2_lr2e4_ep40_rm2_probe:d11_b_k25_probe d10_t2_lr2e4_ep40_s1_probe:d11_b_s2_probe"
NAMES=""
for p in $PAIRS; do NAMES="$NAMES ${p#*:}"; done
for p in $PAIRS $PROBES; do
  grep -q -- "--name ${p#*:} " "$S/run_eleventh_${C}.sh" || { echo "stop: ${p#*:} is not a run of ${C}"; exit 3; }
done
[ "$(grep -c -E '^ +"--name ' "$S/run_eleventh_${C}.sh")" = "12" ] || { echo "stop: ${C} does not have 12 runs"; exit 3; }
LOCAL=""
for p in $PAIRS; do
  LOCAL="$LOCAL ${p%%:*}=$(shasum -a 256 "$B/data/unlearning/candidates/${p%%:*}.pt" | cut -c1-64)"
done
cat > "$S/batch_eleventh.sh" <<WRAP
#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=runs/5ac28b2/tasks/unlearning
bash \$REPO/run_eleventh.sh "0 1 2 3"
code=\$?
if [ ! -s \$OUT/seventh/logits.npz ]; then
  good=""
  for n in $NAMES; do
    if [ -s \$OUT/\$n.pt ]; then good="\$good,\$n"; fi
  done
  CUDA_VISIBLE_DEVICES=0 python \$REPO/dump_logits.py --names "\${good#,}" --views exact,jitter --out \$OUT/seventh/logits.npz > \$OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> \$OUT/seventh/failed.txt
  (cd \$OUT && sha256sum k11_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit \$code
WRAP
bash -n "$S/batch_eleventh.sh"
uv run python scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=begg-ul11 TIME=00:45:00 TASK_DIR=$T "COMMAND=bash output/desk/batch_eleventh.sh" > "$S/batch_eleventh.sbatch"
cat > "$S/stage_eleventh_remote.sh" <<REMOTE
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
for f in seventh/log_0.txt seventh/log_1.txt seventh/log_2.txt seventh/log_3.txt seventh/log_4.txt seventh/log_5.txt seventh/log_6.txt seventh/log_7.txt seventh/log_8.txt seventh/log_9.txt seventh/log_10.txt seventh/log_11.txt seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
  : > \$f
done
cd $T
R=runs/5ac28b2/tasks/unlearning
if [ -e \$R/run_tenth.sh ]; then mv \$R/run_tenth.sh \$R/run_eleventh.sh; fi
if [ -e output/desk/batch_tenth.sh ]; then mv output/desk/batch_tenth.sh output/desk/batch_eleventh.sh; fi
if [ -e output/desk/batch_tenth.sbatch ]; then mv output/desk/batch_tenth.sbatch output/desk/batch_eleventh.sbatch; fi
for f in \$R/run_eleventh.sh output/desk/batch_eleventh.sh output/desk/batch_eleventh.sbatch output/desk/eighth.out; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=\$(find output runs -xdev | wc -l)
echo "files before \$before, after \$after"
test -s output/kkrgp/k7_e3t2_rm.pt && echo "k7_e3t2_rm.pt untouched"
test -s output/kkrgp/xfit_extras.npz && echo "targets file present"
REMOTE
ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 8; }
ssh jureca-mux2 "ls -d $T >/dev/null" || { echo "stop: no scratch on this node"; exit 8; }
ssh jureca-mux2 'bash -s' < "$S/stage_eleventh_remote.sh"
ssh jureca-mux2 "cat > $T/runs/5ac28b2/tasks/unlearning/run_eleventh.sh" < "$S/run_eleventh_${C}.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_eleventh.sh" < "$S/batch_eleventh.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_eleventh.sbatch" < "$S/batch_eleventh.sbatch"
for f in csd.py dump_logits.py lib.py run_eleventh.sh; do
  a=$(git show "${C}:tasks/unlearning/${f}" | shasum -a 256 | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/runs/5ac28b2/tasks/unlearning/${f}" | cut -c1-64)
  [ "$a" = "$b" ] && echo "${f} matches ${C}" || { echo "stop: ${f} on scratch differs from ${C}"; exit 9; }
done
for f in batch_eleventh.sh batch_eleventh.sbatch; do
  a=$(shasum -a 256 "$S/$f" | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/output/desk/$f" | cut -c1-64)
  [ "$a" = "$b" ] && echo "$f matches the rendered copy" || { echo "stop: $f on scratch differs"; exit 9; }
done
ssh jureca-mux2 "cd $T && grep -c -E 'time=00:45:00|batch_eleventh.sh|reservation=cispahack|gres=gpu:4|job-name=begg-ul11' output/desk/batch_eleventh.sbatch && grep -c 'run_eleventh.sh' output/desk/batch_eleventh.sh && ls output/kkrgp/k11_* output/kkrgp/d11_* | wc -l && find output/kkrgp -maxdepth 1 \( -name 'k11_*' -o -name 'd11_*' \) -size +0 | wc -l && ls output/kkrgp/k10_* output/kkrgp/d10_* 2>/dev/null | wc -l && df --output=iavail /p/scratch/training2666 | tail -1"
echo "staged ${C} at $(date +%H:%M:%S)"
