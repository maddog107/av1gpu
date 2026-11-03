#!/bin/bash
# kern_single.sh: k_entropy_win GPU time in single-file mode (nsys), main / br (commit 1) / br2 (both), 3 runs each, on
# the 8 MP photo (16 tiles) and a 1-2 MP photo (one tile).  ~/results/prof/kern_single.txt
O=~/results/prof/kern_single.txt; : > $O
F8=/home/ubuntu/images/0cf2daca3f1c50e1_large.jpg
F1=$(python3 -c "
import glob
from PIL import Image
for p in sorted(glob.glob('/home/ubuntu/images/*.jpg')):
    w, h = Image.open(p).size
    if 1.6e6 <= w * h < 2e6: print(p); break")
cd ~/prof
for f in $F8 $F1; do
  for b in main br br2; do
    bin=~/$b/build/av1gpu
    for k in 1 2 3; do
      nsys profile -t cuda --sample=none --cpuctxsw=none -o ks -f true $bin $f ~/work/ks.avif > /dev/null 2>&1
      nsys stats --report cuda_gpu_kern_sum --format csv --force-overwrite true --force-export true -o ks ks.nsys-rep > /dev/null 2>&1
      t=$(python3 -c "
import csv
for r in csv.DictReader(open('ks_cuda_gpu_kern_sum.csv')):
    if r['Name'].startswith('void av1g_gpu::k_entropy_win'): print(round(float(r['Total Time (ns)']) / 1e6, 2))")
      echo "$(basename $f) $b run $k: k_entropy_win $t ms" >> $O
    done
  done
done
cat $O
