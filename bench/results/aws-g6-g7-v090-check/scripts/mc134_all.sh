#!/bin/bash
# mc134_all.sh: checks9.sh's 8 memcheck jobs again with compute-sanitizer 2026.3 (CUDA 13.4), the sanitizer that matches
# the CUB of the CUDA 13.4 build; outputs compared with the plain runs of checks9.sh (~/work/san/*_ref).
# ~/results/sanitizer134.txt
S=/usr/local/cuda-13.4/bin/compute-sanitizer; L=~/results/san134; W=~/work/san; mkdir -p $L
R=~/results/sanitizer134.txt; echo "BIN B13 ($(sha256sum < ~/bin/B13 | cut -c1-16)); $($S --version | tail -1)" > $R
while IFS='|' read -r n m in o; do
  t=$(date +%s)
  if [ $m = batch ]; then out=$W/${n}_mc134; rm -rf $out
    $S --tool memcheck --leak-check full --error-exitcode 99 ~/bin/B13 --batch $in --outdir $out $o > $L/$n.log 2>&1; rc=$?
    d=0; nf=0; for f in $W/${n}_ref/*; do nf=$((nf + 1)); cmp -s $f $out/$(basename $f) || d=$((d + 1)); done
  else out=$W/${n}_mc134.avif
    $S --tool memcheck --leak-check full --error-exitcode 99 ~/bin/B13 $in $out $o > $L/$n.log 2>&1; rc=$?
    nf=1; d=0; cmp -s $W/${n}_ref.avif $out || d=1
  fi
  echo "memcheck $n [${o:-default}]: rc $rc, $(( $(date +%s) - t )) s, outputs differing from the plain run: $d of $nf; $(grep -aE 'SUMMARY' $L/$n.log | tail -1 | sed 's/=========//')" >> $R
done < $W/jobs.txt
echo SANITIZER134_DONE >> $R
