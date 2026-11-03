#!/bin/bash
# post8.sh (after the speed runs): decoder gate of the PR head on the sets fix_llsets.sh completed (gray 9, mono 47)
# and on 10 single-file-sized photos; nsys of the size range (one pass) for main and the PR head: k_entropy_win time
# and the entropy tail.  ~/results/gate_post.txt, ~/results/prof/nsys_sr_{main,br2}_post.txt
B=~/br2/build/av1gpu; G=~/br2/tests/decoder_gate.py
R=~/results/gate_post.txt; : > $R
gate() { local n=$1; shift; echo "== $n: $*" >> $R; python3 $G --bin $B --jobs 8 "$@" > ~/results/gate_$n.log 2>&1
  echo "rc $?" >> $R; tail -n 2 ~/results/gate_$n.log >> $R; }
gate gray9_q120 --images ~/sets/gray
gate mono47_q120 --images ~/sets/mono
gate misc13_q120 --images ~/ll/misc
echo GATE_DONE >> $R
cd ~/prof
for b in main br2; do
  bin=~/main/build/av1gpu; [ $b = br2 ] && bin=$B
  nsys profile -t cuda --sample=none --cpuctxsw=none -o sr_${b}_post -f true $bin --batch ~/images --outdir ~/work/o_post --stats > sr_${b}_post.out 2>&1
  nsys export --type sqlite -o sr_${b}_post.sqlite -f true sr_${b}_post.nsys-rep > /dev/null 2>&1
  { grep -E "MP/s|stage ms" sr_${b}_post.out; python3 ~/nsys_an.py sr_${b}_post.sqlite | head -12; } > ~/results/prof/nsys_sr_${b}_post.txt
done
echo POST_DONE >> $R
