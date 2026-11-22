set -euo pipefail
cd /p/scratch/training2666/hibti1/Begg/unlearning
[ "$(squeue -u $USER -h -o %i | wc -l)" = "0" ] || { echo "stop: a job of ours is in the queue"; exit 4; }
before=$(find output runs -xdev | wc -l)
cd output/kkrgp
for kv in  k9_e3t3_lr1e4=63c656fdbf241ea9d753c4fe636dea91cadbeaccc75931ebc70cd42679c3bd13 k9_e35t2_lr1e4=877ff5c92a275c36f43375995dfa5d31ad0e78a269bd9a81fb7b7c25ccc0ce57 k9_e3t2_lr2e4=da3c7f29e8b1e221fc598286c0b9e86ddea63a75fe4f8e41cb4b7d67edf15d2d k9_e3t1_lr1e4=22a45bf427c0ae5611eaf4ebbe837b643609cd2709bc4a5f76a6c7f8db797d22 k9_e3t2_lr1e4_ep40=5e3a9b15412f4cc9415d9c38175bdb9fa8de782b83a824280250e196e812d008 k9_e3t2_lr2e4_ep40=989fd0ca1d46c5e2d51bbeeef737d1e67a650cfeab29b83b803a74e22e485ac8; do
  name=${kv%%=*}; want=${kv#*=}
  if [ -e $name.pt ]; then
    have=$(sha256sum $name.pt | cut -c1-64)
    [ "$have" = "$want" ] || { echo "stop: $name.pt differs from the laptop copy"; exit 5; }
  fi
done
for p in k9_e3t3_lr1e4:k10_t2_lr2e4_ep80 k9_e35t2_lr1e4:k10_t2_lr3e4_ep60 k9_e3t2_lr2e4:k10_t2_lr4e4_ep40 k9_e3t1_lr1e4:k10_t15_lr2e4_ep40 k9_e3t2_lr1e4_ep40:k10_t2_lr2e4_ep40_rm2 k9_e3t2_lr2e4_ep40:k10_t2_lr2e4_ep40_s1; do
  old=${p%%:*}; new=${p#*:}
  for ext in pt json; do
    if [ -e $old.$ext ]; then mv $old.$ext $new.$ext; fi
    [ -e $new.$ext ] || { echo "stop: $new.$ext is missing"; exit 6; }
    : > $new.$ext
  done
done
for p in d9_e3t3_lr1e4_probe:d10_t2_lr2e4_ep80_probe d9_e35t2_lr1e4_probe:d10_t2_lr3e4_ep60_probe d9_e3t2_lr2e4_probe:d10_t2_lr4e4_ep40_probe d9_e3t1_lr1e4_probe:d10_t15_lr2e4_ep40_probe d9_e3t2_lr1e4_ep40_probe:d10_t2_lr2e4_ep40_rm2_probe d9_e3t2_lr2e4_ep40_probe:d10_t2_lr2e4_ep40_s1_probe; do
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
if [ -e $R/run_ninth.sh ]; then mv $R/run_ninth.sh $R/run_tenth.sh; fi
if [ -e output/desk/batch_ninth.sh ]; then mv output/desk/batch_ninth.sh output/desk/batch_tenth.sh; fi
if [ -e output/desk/batch_ninth.sbatch ]; then mv output/desk/batch_ninth.sbatch output/desk/batch_tenth.sbatch; fi
for f in $R/run_tenth.sh output/desk/batch_tenth.sh output/desk/batch_tenth.sbatch output/desk/eighth.out; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
done
: > output/desk/eighth.out
after=$(find output runs -xdev | wc -l)
echo "files before $before, after $after"
test -s output/kkrgp/k7_e3t2_rm.pt && echo "k7_e3t2_rm.pt untouched"
test -s output/kkrgp/xfit_extras.npz && echo "targets file present"
