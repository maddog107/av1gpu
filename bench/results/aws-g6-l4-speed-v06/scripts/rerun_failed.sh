#!/bin/bash
# rerun_failed.sh: rerun (alone on the GPU) every identity set whose line in ~/results/identity.txt shows a failed run
# (rc not 0/0) or a difference; results appended to ~/results/identity_rerun.txt
A=~/old/build/av1gpu; B=~/new/build/av1gpu; R=~/results/identity_rerun.txt
echo "rerun $(date -u +%H:%M:%S)" >> $R
grep -E '^[a-z0-9_]+: inputs' ~/results/identity.txt | grep -v '^single_file' | grep -vE 'differ 0, .*rc 0/0' | while read -r l; do
  n=${l%%:*}
  o=$(echo "$l" | sed -E 's/.*opts \[([^]]*)\].*/\1/')
  e=$(echo "$l" | grep -oE 'GP_ENCODERS=[0-9]+')
  src=$(grep -E "^\\\$C $n " ~/identity.sh | awk '{print $3}')
  src=${src/#\~/$HOME}
  env $e bash ~/cmpset.sh $n $src $A $B $o >> $R 2>&1
done
echo RERUN_DONE >> $R
