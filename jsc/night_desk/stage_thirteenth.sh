#!/bin/bash
set -euo pipefail
C=$1
S=SCRATCHPAD
B=$HOME/orca/workspaces/euro-ai-championship-2026/brill
E=$HOME/worktrees/euro-ai-championship-2026/desk-unlearning/tasks/unlearning/reports/21/eleventh
T=/p/scratch/training2666/hibti1/Begg/unlearning
R=runs/5ac28b2/tasks/unlearning
CODE="csd.py lib.py soup.py splice_head.py xfit.py dump_logits.py run_thirteenth.sh"
cd "$B"
git fetch -q origin
mkdir -p "$S/thirteenth"
for f in $CODE; do
  git show "${C}:tasks/unlearning/${f}" > "$S/thirteenth/${f}"
done
bash -n "$S/thirteenth/run_thirteenth.sh"
RENAMES="k11_b_ep20.pt:k13_mix_s0.pt k11_b_ep20.json:k13_mix_s0.json d11_b_ep20_probe.json:d13_mix_probe.json k11_b_bl4.pt:k13_mix_s1.pt k11_b_bl4.json:k13_mix_s1.json d11_b_bl4_probe.json:xfit_extras_s1.json k11_b_rm2.pt:k13_mix_soup_x2.pt k11_b_rm2.json:xfit_extras_s2.json d11_b_rm2_probe.json:xfit_extras_s1_exact.npy k11_b_k25.pt:xfit_extras_s1.npz k11_b_k25.json:xfit_extras_s2.npz d11_b_k25_probe.json:xfit_extras_s2_exact.npy seventh/log_0.txt:seventh/log_xfit_s1.txt seventh/log_1.txt:seventh/log_xfit_s2.txt seventh/log_2.txt:seventh/log_mix_s0.txt seventh/log_3.txt:seventh/log_mix_s1.txt seventh/log_4.txt:seventh/log_mix_probe.txt seventh/log_5.txt:seventh/log_soup.txt"
for p in $RENAMES; do
  old=${p%%:*}
  case "$old" in
    *.pt) ;;
    *) [ -s "$E/${old#seventh/}" ] || { echo "stop: ${old} is not in the repository's eleventh reports"; exit 3; } ;;
  esac
done
for f in log_6.txt log_dump.txt logits.npz sha256.txt slurm.out; do
  [ -s "$E/$f" ] || { echo "stop: $f is not in the repository's eleventh reports"; exit 3; }
done
for n in k13_mix_s0 k13_mix_s1 d13_mix_probe; do
  grep -q -- "--name ${n} " "$S/thirteenth/run_thirteenth.sh" || { echo "stop: ${n} is not a run of ${C}"; exit 3; }
done
for n in xfit_extras_s1.npz xfit_extras_s2.npz k13_mix_soup_x2.pt log_xfit_s1.txt log_xfit_s2.txt log_mix_s0.txt log_mix_s1.txt log_mix_probe.txt log_soup.txt log_dump.txt logits.npz sha256.txt failed.txt; do
  grep -q -- "${n}" "$S/thirteenth/run_thirteenth.sh" || { echo "stop: ${n} is not written by ${C}"; exit 3; }
done
LOCAL=""
for n in k11_b_ep20 k11_b_bl4 k11_b_rm2 k11_b_k25; do
  LOCAL="$LOCAL ${n}=$(shasum -a 256 "$B/data/unlearning/candidates/${n}.pt" | cut -c1-64)"
done
cat > "$S/batch_thirteenth.sh" <<WRAP
#!/bin/bash
set -uo pipefail
OUT=output/kkrgp
REPO=$R
bash \$REPO/run_thirteenth.sh "0 1 2 3"
code=\$?
if [ ! -s \$OUT/seventh/logits.npz ]; then
  good=""
  for n in k13_mix_s0 k13_mix_s1 k13_mix_soup_x2; do
    if [ -s \$OUT/\$n.pt ]; then good="\$good,\$n"; fi
  done
  if [ -n "\$good" ]; then
    CUDA_VISIBLE_DEVICES=0 python \$REPO/dump_logits.py --names "\${good#,}" --views exact,jitter --out \$OUT/seventh/logits.npz > \$OUT/seventh/log_dump.txt 2>&1 || echo "second dump failed" >> \$OUT/seventh/failed.txt
  fi
  (cd \$OUT && sha256sum k13_*.pt > seventh/sha256.txt 2>/dev/null)
fi
exit \$code
WRAP
bash -n "$S/batch_thirteenth.sh"
uv run python scripts/render_template.py jsc/jureca_gpu.sbatch.template JOB_NAME=begg-ul13 TIME=00:45:00 TASK_DIR=$T "COMMAND=bash output/desk/batch_thirteenth.sh" > "$S/batch_thirteenth.sbatch"
cat > "$S/stage_thirteenth_remote.sh" <<REMOTE
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
for f in seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
  : > \$f
done
cd $T
if [ -e output/kkrgp/seventh/log_6.txt ] && [ ! -e $R/soup.py ]; then mv output/kkrgp/seventh/log_6.txt $R/soup.py; fi
if [ -e $R/run_eleventh.sh ]; then mv $R/run_eleventh.sh $R/run_thirteenth.sh; fi
if [ -e output/desk/batch_eleventh.sh ]; then mv output/desk/batch_eleventh.sh output/desk/batch_thirteenth.sh; fi
if [ -e output/desk/batch_eleventh.sbatch ]; then mv output/desk/batch_eleventh.sbatch output/desk/batch_thirteenth.sbatch; fi
for f in $R/soup.py $R/run_thirteenth.sh $R/csd.py $R/lib.py $R/splice_head.py $R/xfit.py $R/dump_logits.py output/desk/batch_thirteenth.sh output/desk/batch_thirteenth.sbatch output/desk/eighth.out; do
  [ -e \$f ] || { echo "stop: \$f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=\$(find output runs -xdev | wc -l)
echo "files before \$before, after \$after"
for keep in k8_e3t2_rm1_lr1e4.pt k11_b_s1.pt k11_b_s2.pt xfit_extras.npz xfit_extras_exact.npy features.npz; do
  test -s output/kkrgp/\$keep && echo "\$keep untouched"
done
REMOTE
ssh -O check jureca03 >/dev/null 2>&1 || { echo "stop: the link to jureca03 is down"; exit 8; }
ssh jureca-mux2 "ls -d $T >/dev/null" || { echo "stop: no scratch on this node"; exit 8; }
ssh jureca-mux2 'bash -s' < "$S/stage_thirteenth_remote.sh"
for f in csd.py lib.py soup.py run_thirteenth.sh; do
  ssh jureca-mux2 "cat > $T/$R/${f}" < "$S/thirteenth/${f}"
done
b=$(ssh jureca-mux2 "sha256sum $T/$R/splice_head.py" | cut -c1-64)
a=$(shasum -a 256 "$S/thirteenth/splice_head.py" | cut -c1-64)
if [ "$a" != "$b" ]; then
  echo "splice_head.py on scratch is older; replacing it in place"
  ssh jureca-mux2 "cat > $T/$R/splice_head.py" < "$S/thirteenth/splice_head.py"
fi
ssh jureca-mux2 "cat > $T/output/desk/batch_thirteenth.sh" < "$S/batch_thirteenth.sh"
ssh jureca-mux2 "cat > $T/output/desk/batch_thirteenth.sbatch" < "$S/batch_thirteenth.sbatch"
for f in $CODE; do
  a=$(shasum -a 256 "$S/thirteenth/${f}" | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/$R/${f}" | cut -c1-64)
  [ "$a" = "$b" ] && echo "${f} matches ${C}" || { echo "stop: ${f} on scratch differs from ${C}"; exit 9; }
done
for f in batch_thirteenth.sh batch_thirteenth.sbatch; do
  a=$(shasum -a 256 "$S/$f" | cut -c1-64)
  b=$(ssh jureca-mux2 "sha256sum $T/output/desk/$f" | cut -c1-64)
  [ "$a" = "$b" ] && echo "$f matches the rendered copy" || { echo "stop: $f on scratch differs"; exit 9; }
done
ssh jureca-mux2 "cd $T && grep -c -E 'time=00:45:00|batch_thirteenth.sh|reservation=cispahack|gres=gpu:4|job-name=begg-ul13' output/desk/batch_thirteenth.sbatch && grep -c 'run_thirteenth.sh' output/desk/batch_thirteenth.sh && ls output/kkrgp/k13_* output/kkrgp/d13_* output/kkrgp/xfit_extras_s* | wc -l && find output/kkrgp -maxdepth 1 \( -name 'k13_*' -o -name 'd13_*' -o -name 'xfit_extras_s*' \) -size +0 | wc -l && ls output/kkrgp/seventh | tr '\n' ' ' && df --output=iavail /p/scratch/training2666 | tail -1"
echo "staged ${C} at $(date +%H:%M:%S)"
