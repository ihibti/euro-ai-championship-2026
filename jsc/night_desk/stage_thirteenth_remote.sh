set -euo pipefail
cd /p/scratch/training2666/hibti1/Begg/unlearning
[ "$(squeue -u $USER -h -o %i | wc -l)" = "0" ] || { echo "stop: a job of ours is in the queue"; exit 4; }
before=$(find output runs -xdev | wc -l)
cd output/kkrgp
for kv in  k11_b_ep20=28870e41aefc6297aa42363396ad5f7a21d3ddff9122ce4eef9378052aad6c08 k11_b_bl4=d3445e8d2f6b70a980425818a376b5f0ba1a69f8218d90448292c8afce90bdfc k11_b_rm2=6634dca66a9c3171c7549a620cf8fe01b73587fe75425b1d6b4a2efcdc8c3293 k11_b_k25=185927b94eb415d14a6fe742d687be4d76a5b93d458c4049a853dae277501ff0; do
  name=${kv%%=*}; want=${kv#*=}
  if [ -e $name.pt ]; then
    have=$(sha256sum $name.pt | cut -c1-64)
    [ "$have" = "$want" ] || { echo "stop: $name.pt differs from the laptop copy"; exit 5; }
  fi
done
for p in k11_b_ep20.pt:k13_mix_s0.pt k11_b_ep20.json:k13_mix_s0.json d11_b_ep20_probe.json:d13_mix_probe.json k11_b_bl4.pt:k13_mix_s1.pt k11_b_bl4.json:k13_mix_s1.json d11_b_bl4_probe.json:xfit_extras_s1.json k11_b_rm2.pt:k13_mix_soup_x2.pt k11_b_rm2.json:xfit_extras_s2.json d11_b_rm2_probe.json:xfit_extras_s1_exact.npy k11_b_k25.pt:xfit_extras_s1.npz k11_b_k25.json:xfit_extras_s2.npz d11_b_k25_probe.json:xfit_extras_s2_exact.npy seventh/log_0.txt:seventh/log_xfit_s1.txt seventh/log_1.txt:seventh/log_xfit_s2.txt seventh/log_2.txt:seventh/log_mix_s0.txt seventh/log_3.txt:seventh/log_mix_s1.txt seventh/log_4.txt:seventh/log_mix_probe.txt seventh/log_5.txt:seventh/log_soup.txt; do
  old=${p%%:*}; new=${p#*:}
  if [ -e $old ]; then mv $old $new; fi
  [ -e $new ] || { echo "stop: $new is missing"; exit 6; }
  : > $new
done
for f in seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
  : > $f
done
cd /p/scratch/training2666/hibti1/Begg/unlearning
if [ -e output/kkrgp/seventh/log_6.txt ] && [ ! -e runs/5ac28b2/tasks/unlearning/soup.py ]; then mv output/kkrgp/seventh/log_6.txt runs/5ac28b2/tasks/unlearning/soup.py; fi
if [ -e runs/5ac28b2/tasks/unlearning/run_eleventh.sh ]; then mv runs/5ac28b2/tasks/unlearning/run_eleventh.sh runs/5ac28b2/tasks/unlearning/run_thirteenth.sh; fi
if [ -e output/desk/batch_eleventh.sh ]; then mv output/desk/batch_eleventh.sh output/desk/batch_thirteenth.sh; fi
if [ -e output/desk/batch_eleventh.sbatch ]; then mv output/desk/batch_eleventh.sbatch output/desk/batch_thirteenth.sbatch; fi
for f in runs/5ac28b2/tasks/unlearning/soup.py runs/5ac28b2/tasks/unlearning/run_thirteenth.sh runs/5ac28b2/tasks/unlearning/csd.py runs/5ac28b2/tasks/unlearning/lib.py runs/5ac28b2/tasks/unlearning/splice_head.py runs/5ac28b2/tasks/unlearning/xfit.py runs/5ac28b2/tasks/unlearning/dump_logits.py output/desk/batch_thirteenth.sh output/desk/batch_thirteenth.sbatch output/desk/eighth.out; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=$(find output runs -xdev | wc -l)
echo "files before $before, after $after"
for keep in k8_e3t2_rm1_lr1e4.pt k11_b_s1.pt k11_b_s2.pt xfit_extras.npz xfit_extras_exact.npy features.npz; do
  test -s output/kkrgp/$keep && echo "$keep untouched"
done
