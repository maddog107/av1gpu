#!/bin/bash
# mc134.sh: the lead's two checks of the Blackwell memcheck report.
#  1. compute-sanitizer 2026.3 (CUDA 13.4, apt cuda-sanitizer-13-4) memcheck on photo.jpg and the production pair with
#     B13 and A13 (and 2026.1 again for reference);
#  2. a standalone repro built by nvcc 13.4 (CUB 3.4.3, as the 13.4 package) for sm_120: cubscan.cu (9 sizes) and
#     cubsweep.cu (gpu_entropy.cu:1734's call for every n up to 200,000) under both sanitizers.
# ~/results/mc134.txt
R=~/results/mc134.txt; : > $R; L=~/results/mc134; mkdir -p $L
S132=/usr/local/cuda-13.2/bin/compute-sanitizer; S134=/usr/local/cuda-13.4/bin/compute-sanitizer
for S in $S134 $S132; do echo "sanitizer $S: $($S --version | tail -1)" >> $R; done
one() {   # tag sanitizer cmd...
  local t=$1 S=$2; shift 2
  $S --tool memcheck --error-exitcode 99 --print-limit 5 "$@" > $L/$t.log 2>&1; local rc=$?
  echo "$t: rc $rc; $(grep -c 'Invalid __shared__' $L/$t.log) invalid shared reads; first: $(grep -m1 '^=========     at ' $L/$t.log | sed 's/+0x[0-9a-f]*//; s/=========     //' | cut -c1-90); $(grep -a 'ERROR SUMMARY' $L/$t.log | tail -1 | sed 's/=========//'); $(grep -aE '^(CUB|n [0-9])' $L/$t.log | tail -1)" >> $R
}
for b in B13 A13; do
  one san134_${b}_photo $S134 ~/bin/$b ~/st/photo.jpg /tmp/m134_$b.avif
  rm -rf /tmp/m134p_$b; one san134_${b}_pair $S134 ~/bin/$b --batch ~/st/pair.txt --outdir /tmp/m134p_$b
done
N=/usr/local/cuda-13.4/bin/nvcc
$N -O2 -arch=sm_120 -o ~/cubscan_134 ~/cubscan.cu >> $R 2>&1
$N -O2 -arch=sm_120 -o ~/cubsweep_134 ~/cubsweep.cu >> $R 2>&1
for S in $S134 $S132; do
  v=$(basename $(dirname $(dirname $S)))
  one cubscan_${v} $S ~/cubscan_134
  one cubsweep_${v}_1_5000 $S ~/cubsweep_134 1 5000
  one cubsweep_${v}_5001_200000_7 $S ~/cubsweep_134 5001 200000 7
done
echo MC134_DONE >> $R
