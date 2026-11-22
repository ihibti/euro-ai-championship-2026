set -euo pipefail
cd /p/scratch/training2666/hibti1/Begg/unlearning
[ "$(squeue -u $USER -h -o %i | wc -l)" = "0" ] || { echo "stop: a job of ours is in the queue"; exit 4; }
before=$(find output runs -xdev | wc -l)
cd output/kkrgp
for kv in  k7_e3t2=2b397865089c32b5fbc38ee190657f11556e4eeff33a8266d21b0d3284a2a428 k7_e3t2_rm3=222348bf66d7aee3768b8fbc29d4587c972290e76636bd2b217f5fac29dd15ab k7_e3t2_b3_rm3=ed2d40645e3716d66895dd1e3d024861139229d205c28777d7825659f4d2fca9 k7_e3t2_ex_rm3=326d1677cd9d9a53f49a386553015c95080eeb127c54de097ac4d32f5b51e599 k7_e2t2_rm3=0f357d783501d88e39da7097d0e765b942c546f0ad9b06a5d5dfaf99a7aaa17e u7_t32_rm3=e488346e0c54c6d69df5b6e128f5e951896bfe1d3388d3caa17dcdb4ede6456f; do
  name=${kv%%=*}; want=${kv#*=}
  if [ -e $name.pt ]; then
    have=$(sha256sum $name.pt | cut -c1-64)
    [ "$have" = "$want" ] || { echo "stop: $name.pt differs from the laptop copy"; exit 5; }
  fi
done
for p in k7_e3t2:k9_e3t2_lr2e4 k7_e3t2_rm3:k9_e3t2_lr1e4_ep40 k7_e3t2_b3_rm3:k9_e3t2_lr2e4_ep40 k7_e3t2_ex_rm3:k9_e3t3_lr1e4 k7_e2t2_rm3:k9_e3t1_lr1e4 u7_t32_rm3:k9_e35t2_lr1e4; do
  old=${p%%:*}; new=${p#*:}
  for ext in pt json; do
    if [ -e $old.$ext ]; then mv $old.$ext $new.$ext; fi
    [ -e $new.$ext ] || { echo "stop: $new.$ext is missing"; exit 6; }
    : > $new.$ext
  done
done
for p in d7_e3t2_probe:d9_e3t2_lr2e4_probe d7_e3t2_rm3_probe:d9_e3t2_lr1e4_ep40_probe d7_e3t2_b3_rm3_probe:d9_e3t2_lr2e4_ep40_probe d7_e3t2_ex_rm3_probe:d9_e3t3_lr1e4_probe d7_e2t2_rm3_probe:d9_e3t1_lr1e4_probe d7_t32_rm3_probe:d9_e35t2_lr1e4_probe; do
  old=${p%%:*}; new=${p#*:}
  if [ -e $old.json ]; then mv $old.json $new.json; fi
  [ -e $new.json ] || { echo "stop: $new.json is missing"; exit 6; }
  : > $new.json
done
if [ ! -e seventh/failed.txt ]; then mv eighth/log_15.txt seventh/failed.txt; fi
for f in seventh/log_0.txt seventh/log_1.txt seventh/log_2.txt seventh/log_3.txt seventh/log_4.txt seventh/log_5.txt seventh/log_6.txt seventh/log_7.txt seventh/log_8.txt seventh/log_9.txt seventh/log_10.txt seventh/log_11.txt seventh/log_dump.txt seventh/logits.npz seventh/sha256.txt seventh/failed.txt; do
  [ -e $f ] || { echo "stop: $f is missing"; exit 7; }
  : > $f
done
cd /p/scratch/training2666/hibti1/Begg/unlearning
R=runs/5ac28b2/tasks/unlearning
if [ -e $R/run_eighth.sh ]; then mv $R/run_eighth.sh $R/run_ninth.sh; fi
if [ -e output/desk/batch_5ac28b2.sh ]; then mv output/desk/batch_5ac28b2.sh output/desk/batch_ninth.sh; fi
if [ -e output/desk/batch_5ac28b2.sbatch ]; then mv output/desk/batch_5ac28b2.sbatch output/desk/batch_ninth.sbatch; fi
: > output/desk/eighth.out
after=$(find output runs -xdev | wc -l)
echo "files before $before, after $after"
test -e output/kkrgp/k7_e3t2_rm.pt && test -s output/kkrgp/k7_e3t2_rm.pt && echo "k7_e3t2_rm.pt untouched"
test -s output/kkrgp/xfit_extras.npz && echo "targets file present"
