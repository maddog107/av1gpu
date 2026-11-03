#!/bin/bash
# mc_ab.sh: compute-sanitizer memcheck on photo.jpg (single-file, default) and the production pair (--batch) with each
# binary: B13 (v0.9.0 CUDA 13.4), A13 (v0.8.0 CUDA 13.4), B12 (v0.9.0 CUDA 12.9), A12 (v0.8.0 CUDA 12.9).
# Does the CUB DeviceScan report on Blackwell come with v0.9.0, or with the CUDA 13.4 toolkit?  ~/results/mc_ab.txt
CS=/usr/local/cuda/bin/compute-sanitizer; L=~/results/mc_ab; mkdir -p $L; R=~/results/mc_ab.txt; : > $R
for b in B13 A13 B12 A12; do
  for t in photo pair; do
    if [ $t = photo ]; then a="$HOME/st/photo.jpg /tmp/mcab_$b.avif"; else rm -rf /tmp/mcab_${b}_pair; a="--batch $HOME/st/pair.txt --outdir /tmp/mcab_${b}_pair"; fi
    $CS --tool memcheck --error-exitcode 99 ~/bin/$b $a > $L/${b}_$t.log 2>&1; rc=$?
    echo "$b $t: rc $rc; $(grep -c 'Invalid __shared__ read' $L/${b}_$t.log) invalid shared reads, first at: $(grep -m1 '^=========     at ' $L/${b}_$t.log | sed 's/+0x[0-9a-f]*//' | cut -c1-120); $(grep -a 'ERROR SUMMARY' $L/${b}_$t.log | tail -1 | sed 's/=========//')" >> $R
  done
done
echo MC_AB_DONE >> $R
