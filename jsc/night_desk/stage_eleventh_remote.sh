set -euo pipefail
cd /p/scratch/training2666/hibti1/Begg/unlearning
[ "$(squeue -u $USER -h -o %i | wc -l)" = "0" ] || { echo "stop: a job of ours is in the queue"; exit 4; }
before=$(find output runs -xdev | wc -l)
cd output/kkrgp
for kv in  k10_t2_lr2e4_ep80=7886100a211e76bcc29925877c3bd89ccdf2622ebc7889f07ffd38aa7f593a29 k10_t2_lr3e4_ep60=9d898a4da983e90d6ee53e89c6029f0572b023ebf45c7e11de6c78934c71b2cf k10_t2_lr4e4_ep40=bdde2addc359157c56e37d706831d32838aa2fb01db73f12b402dac13151ec66 k10_t15_lr2e4_ep40=894c64156c1da042d429ce869ae60ac33c0bea48ffa4097edced8fa345f89efd k10_t2_lr2e4_ep40_rm2=da560bad035c9f3952a9f655cfba0be60d0dd5da34b33c1b7270073673a46e7d k10_t2_lr2e4_ep40_s1=9d8cee4acaf18583c5df6295770cd5bd690d569924b6435062042dfa7ecee033; do
  name=${kv%%=*}; want=${kv#*=}
  if [ -e $name.pt ]; then
    have=$(sha256sum $name.pt | cut -c1-64)
    [ "$have" = "$want" ] || { echo "stop: $name.pt differs from the laptop copy"; exit 5; }
  fi
done
for p in k10_t2_lr2e4_ep80:k11_b_s1 k10_t2_lr3e4_ep60:k11_b_ep20 k10_t2_lr4e4_ep40:k11_b_rm2 k10_t15_lr2e4_ep40:k11_b_bl4 k10_t2_lr2e4_ep40_rm2:k11_b_k25 k10_t2_lr2e4_ep40_s1:k11_b_s2; do
  old=${p%%:*}; new=${p#*:}
  for ext in pt json; do
    if [ -e $old.$ext ]; then mv $old.$ext $new.$ext; fi
    [ -e $new.$ext ] || { echo "stop: $new.$ext is missing"; exit 6; }
    : > $new.$ext
  done
done
for p in d10_t2_lr2e4_ep80_probe:d11_b_s1_probe d10_t2_lr3e4_ep60_probe:d11_b_ep20_probe d10_t2_lr4e4_ep40_probe:d11_b_rm2_probe d10_t15_lr2e4_ep40_probe:d11_b_bl4_probe d10_t2_lr2e4_ep40_rm2_probe:d11_b_k25_probe d10_t2_lr2e4_ep40_s1_probe:d11_b_s2_probe; do
  old=${p%%:*}; new=${p#*:}
  if [ -e $old.json ]; then mv $old.json $new.json; fi
  [ -e $new.json ] || { echo "stop: $new.json is missing"; exit 6; }
  : > $new.json
done
for f in seventh/log_0.txt seventh/log_1.txt seventh/log_2.txt seventh/log_3.txt seventh/log_4.txt seventh/log_5.txt seventh/log_6.txt seventh/log_7.txt seventh/log_8.txt seventh/log_9.txt seventh/log_10.txt seventh/log_11.txt seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
  : > $f
done
cd /p/scratch/training2666/hibti1/Begg/unlearning
R=runs/5ac28b2/tasks/unlearning
if [ -e $R/run_tenth.sh ]; then mv $R/run_tenth.sh $R/run_eleventh.sh; fi
if [ -e output/desk/batch_tenth.sh ]; then mv output/desk/batch_tenth.sh output/desk/batch_eleventh.sh; fi
if [ -e output/desk/batch_tenth.sbatch ]; then mv output/desk/batch_tenth.sbatch output/desk/batch_eleventh.sbatch; fi
for f in $R/run_eleventh.sh output/desk/batch_eleventh.sh output/desk/batch_eleventh.sbatch output/desk/eighth.out; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=$(find output runs -xdev | wc -l)
echo "files before $before, after $after"
test -s output/kkrgp/k7_e3t2_rm.pt && echo "k7_e3t2_rm.pt untouched"
test -s output/kkrgp/xfit_extras.npz && echo "targets file present"
