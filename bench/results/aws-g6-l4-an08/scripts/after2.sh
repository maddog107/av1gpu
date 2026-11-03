#!/bin/bash
# after2.sh: once verify2.sh is done (speed runs finished), rerun alone on the GPU every identity set of
# identity_new2.txt / identity_pal_new2.txt whose run failed (rc not 0/0: out of GPU memory while the sanitizer runs
# shared the GPU) or that differed -> ~/results/identity_rerun_new2.txt; then memcheck / synccheck again alone
# (sanitizer.sh, tag _new2b)
until grep -q VERIFY2_DONE ~/results/speed2.log 2>/dev/null; do sleep 15; done
A=~/old/build/av1gpu; B=~/new2/build/av1gpu; R=~/results/identity_rerun_new2.txt
echo "rerun $(date -u +%H:%M:%S)" >> $R
cat ~/results/identity_new2.txt ~/results/identity_pal_new2.txt | grep -E '^[a-z0-9_]+: inputs' | grep -v '^single_file' |
  grep -vE 'differ 0, .*rc 0/0' | while read -r l; do
  n=${l%%:*}
  o=$(echo "$l" | sed -E 's/.*opts \[([^]]*)\].*/\1/')
  e=$(echo "$l" | grep -oE 'GP_ENCODERS=[0-9]+')
  if [[ $n == pal_* ]]; then s=${n#pal_}; for k in screen flat_png flat_jpg; do [[ $s == ${k}_* ]] && src=$HOME/pal/sets/$k; done
  else src=$(grep -E "^\\\$C $n " ~/identity.sh | awk '{print $3}'); src=${src/#\~/$HOME}; fi
  env $e bash ~/cmpset.sh $n $src $A $B $o >> $R 2>&1
done
echo RERUN_DONE >> $R
bash ~/sanitizer.sh $B _new2b > /dev/null 2>&1
echo AFTER2_DONE >> $R
