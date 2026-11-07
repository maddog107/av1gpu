#!/bin/bash
# the cw > 2 edge: 4:2:0 JPEGs 1..8 pixels wide (chroma 1..4 samples), tone / gray / trim under the sanitizers, then the model at those widths
CS=/usr/local/cuda/bin/compute-sanitizer; F=$HOME/fix3/build/av1gpu; R=~/results/cwedge; mkdir -p $R; D=/dev/shm/cwe; rm -rf $D; mkdir -p $D/j $D/o
python3 - <<'PY'
import numpy as np
from PIL import Image
rng=np.random.default_rng(4)
for w in range(1,9):
    for h in (1,2,3,7,16,33):
        a=rng.integers(0,256,(h,w,3),dtype=np.uint8); a[:, :w//2]=(250,20,20)
        Image.fromarray(a).save(f'/dev/shm/cwe/j/w{w}h{h}.jpg',quality=95,subsampling=2)
PY
echo "$(ls $D/j | wc -l) JPEGs"
for tool in memcheck racecheck synccheck; do
  for o in "--negate" "--level 10%,90%,1.3 --gamma 0.8" "--auto-level" "--grayscale Lightness" "--colorspace Gray" "--trim --fuzz 3%" "--depth 10 --negate --grayscale Average" "--crop 4x4+0+0 --grayscale RMS" "--crop 3x3+1+1 --colorspace Gray"; do
    rm -rf $D/o; mkdir -p $D/o
    timeout 900 $CS --tool $tool --error-exitcode 99 --print-limit 10 $F --batch $D/j --outdir $D/o --small-preset none --preset turbo3 $o > $R/$tool.$(echo $o | tr -c 'a-z0-9' _).txt 2>&1; rc=$?
    echo "$tool [$o] rc=$rc files=$(ls $D/o | wc -l): $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $R/$tool.$(echo $o | tr -c 'a-z0-9' _).txt | tail -1)"
  done
done
grep -ohE "(k_[a-z0-9_]+|win_group)[<(]" $R/racecheck.*.txt | sort | uniq -c
echo CWEDGE_DONE
