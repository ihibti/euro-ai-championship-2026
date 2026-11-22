#!/bin/bash
set -euo pipefail
C=$1
S=SCRATCHPAD
B=$HOME/orca/workspaces/euro-ai-championship-2026/brill
E=$HOME/worktrees/euro-ai-championship-2026/desk-unlearning/tasks/unlearning/reports/21/thirteenth
T=/p/scratch/training2666/hibti1/Begg/unlearning
R=runs/5ac28b2/tasks/unlearning
CODE="csd.py lib.py dump_logits.py run_fourteenth.sh"
cd "$B"
git fetch -q origin
mkdir -p "$S/fourteenth"
for f in $CODE; do
  git show "${C}:tasks/unlearning/${f}" > "$S/fourteenth/${f}"
done
bash -n "$S/fourteenth/run_fourteenth.sh"
RENAMES="k13_mix_s0.pt:k14_t15_lr1e4.pt k13_mix_s0.json:k14_t15_lr1e4.json k13_mix_s1.pt:k14_t15_lr8e5.pt k13_mix_s1.json:k14_t15_lr8e5.json k13_mix_soup_x2.pt:k14_t1_lr9e5.pt d13_mix_probe.json:k14_t1_lr9e5.json xfit_extras_s1.json:k14_t1_lr75.json xfit_extras_s2.json:d14_t15_lr1e4_probe.json xfit_extras_s1_exact.npy:d14_t15_lr8e5_probe.json xfit_extras_s2_exact.npy:d14_t1_lr9e5_probe.json seventh/log_xfit_s1.txt:seventh/log_14_0.txt seventh/log_xfit_s2.txt:seventh/log_14_1.txt seventh/log_mix_s0.txt:seventh/log_14_2.txt seventh/log_mix_s1.txt:seventh/log_14_3.txt seventh/log_mix_probe.txt:seventh/log_14_4.txt seventh/log_soup.txt:seventh/log_14_5.txt"
NEW="k14_t1_lr75.pt d14_t1_lr75_probe.json seventh/log_14_6.txt seventh/log_14_7.txt"
for p in $RENAMES; do
  old=${p%%:*}
  case "$old" in
    *.pt) ;;
    *) [ -s "$E/${old#seventh/}" ] || { echo "stop: ${old} is not in the repository's thirteenth reports"; exit 3; } ;;
  esac
done
for f in log_dump.txt logits.npz sha256.txt slurm.out; do
  [ -s "$E/$f" ] || { echo "stop: $f is not in the repository's thirteenth reports"; exit 3; }
done
(cd "$HOME/worktrees/euro-ai-championship-2026/desk-unlearning" && [ -z "$(git status --short tasks/unlearning/reports/21/thirteenth)" ] && [ "$(git rev-list --count @{u}..HEAD)" = "0" ]) || { echo "stop: the thirteenth reports are not all committed and pushed"; exit 3; }
for n in k14_t15_lr1e4 k14_t15_lr8e5 k14_t1_lr9e5 k14_t1_lr75 d14_t15_lr1e4_probe d14_t15_lr8e5_probe d14_t1_lr9e5_probe d14_t1_lr75_probe; do
  grep -q -- "--name ${n} " "$S/fourteenth/run_fourteenth.sh" || { echo "stop: ${n} is not a run of ${C}"; exit 3; }
done
for n in log_14_ log_dump.txt logits.npz sha256.txt failed.txt; do
  grep -q -- "${n}" "$S/fourteenth/run_fourteenth.sh" || { echo "stop: ${n} is not written by ${C}"; exit 3; }
done
LOCAL=""
for n in k13_mix_s0 k13_mix_s1 k13_mix_soup_x2; do
  LOCAL="$LOCAL ${n}=$(shasum -a 256 "$B/data/unlearning/candidates/${n}.pt" | cut -c1-64)"
done
cat > "$S/batch_fourteenth.sh" <<WRAP
#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=$R
bash \$REPO/run_fourteenth.sh "0 1 2 3"
code=\$?
if [ ! -s \$OUT/seventh/logits.npz ]; then
  good=""
  for n in k14_t15_lr1e4 k14_t15_lr8e5 k14_t1_lr9e5 k14_t1_lr75; do
    if [ -s \$OUT/\$n.pt ]; then good="\$good,\$n"; fi
  done
  if [ -n "\$good" ]; then
    CUDA_VISIBLE_DEVICES=0 python \$REPO/dump_logits.py --names "\${good#,}" --views exact,jitter --out \$OUT/seventh/logits.npz > \$OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> \$OUT/seventh/failed.txt
  fi
  (cd \$OUT && sha256sum k14_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit \$code
WRAP
bash -n "$S/batch_fourteenth.sh"
uv run python scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=begg-ul14 TIME=00:45:00 TASK_DIR=$T "COMMAND=bash output/desk/batch_fourteenth.sh" > "$S/batch_fourteenth.sbatch"
cat > "$S/stage_fourteenth_remote.sh" <<REMOTE
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
for p in $RENAMES; do
  old=\${p%%:*}; new=\${p#*:}
  if [ -e \$old ]; then mv \$old \$new; fi
  [ -e \$new ] || { echo "stop: \$new is missing"; exit 6; }
  : > \$new
done
for f in $NEW; do
  : > \$f
done
for f in seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
  : > \$f
done
cd $T
if [ -e $R/run_thirteenth.sh ]; then mv $R/run_thirteenth.sh $R/run_fourteenth.sh; fi
if [ -e output/desk/batch_thirteenth.sh ]; then mv output/desk/batch_thirteenth.sh output/desk/batch_fourteenth.sh; fi
if [ -e output/desk/batch_thirteenth.sbatch ]; then mv output/desk/batch_thirteenth.sbatch output/desk/batch_fourteenth.sbatch; fi
for f in $R/run_fourteenth.sh $R/csd.py $R/lib.py $R/dump_logits.py output/desk/batch_fourteenth.sh output/desk/batch_fourteenth.sbatch output/desk/eighth.out; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=\$(find output runs -xdev | wc -l)
echo "files before \$before, after \$after"
for keep in k8_e3t2_rm1_lr1e4.pt k11_b_s1.pt k11_b_s2.pt xfit_extras.npz xfit_extras_exact.npy xfit_extras_s1.npz xfit_extras_s2.npz features.npz; do
  test -s output/kkrgp/\$keep && echo "\$keep untouched"
done
REMOTE
ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 8; }
ssh jureca-mux2 "ls -d $T >/dev/null" || { echo "stop: no scratch on this node"; exit 8; }
ssh jureca-mux2 'bash -s' < "$S/stage_fourteenth_remote.sh"
ssh jureca-mux2 "cat > $T/$R/run_fourteenth.sh" < "$S/fourteenth/run_fourteenth.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_fourteenth.sh" < "$S/batch_fourteenth.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_fourteenth.sbatch" < "$S/batch_fourteenth.sbatch"
for f in $CODE; do
  a=$(shasum -a 256 "$S/fourteenth/${f}" | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/$R/${f}" | cut -c1-64)
  [ "$a" = "$b" ] && echo "${f} matches ${C}" || { echo "stop: ${f} on scratch differs from ${C}"; exit 9; }
done
for f in batch_fourteenth.sh batch_fourteenth.sbatch; do
  a=$(shasum -a 256 "$S/$f" | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/output/desk/$f" | cut -c1-64)
  [ "$a" = "$b" ] && echo "$f matches the rendered copy" || { echo "stop: $f on scratch differs"; exit 9; }
done
ssh jureca-mux2 "cd $T && grep -c -E 'time=00:45:00|batch_fourteenth.sh|reservation=cispahack|gres=gpu:4|job-name=begg-ul14' output/desk/batch_fourteenth.sbatch && grep -c 'run_fourteenth.sh' output/desk/batch_fourteenth.sh && ls output/kkrgp/k14_* output/kkrgp/d14_* | wc -l && find output/kkrgp -maxdepth 1 \( -name 'k14_*' -o -name 'd14_*' \) -size +0 | wc -l && ls output/kkrgp/seventh | tr '\n' ' ' && df --output=iavail /p/scratch/training2666 | tail -1"
echo "staged ${C} at $(date +%H:%M:%S)"
