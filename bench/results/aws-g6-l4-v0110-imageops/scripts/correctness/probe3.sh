#!/bin/bash
source ~/t/try.sh
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
a=np.full((65535,16),255,np.uint8); a[20000:40000,4:12]=60
wpng('/home/ubuntu/adv/tall/gray_16x65535.png',a)
a=np.full((65472,16,3),255,np.uint8); a[20000:40000,4:12]=(60,20,200)
wpng('/home/ubuntu/adv/tall/t_16x65472.png',a)
a=np.full((65473,16,3),255,np.uint8); a[20000:40000,4:12]=(60,20,200)
wpng('/home/ubuntu/adv/tall/t_16x65473.png',a)
PY
echo "== main at height 65535"
try $M ~/adv/tall/t_16x65535.png $T/o.avif --preset turbo3 --yuv 444
try $M ~/adv/tall/t_16x65535.png $T/o.avif --preset turbo3 --lossless
try $M ~/adv/tall/gray_16x65535.png $T/o.avif --preset turbo3
try $M ~/adv/tall/t_16x65535.png $T/o.avif --preset turbo3 --yuv 444 --depth 10
echo "== PR without options at 65535"
try $B ~/adv/tall/t_16x65535.png $T/o.avif --preset turbo3 --yuv 444
try $B ~/adv/tall/gray_16x65535.png $T/o.avif --preset turbo3
echo "== PR options near the limit"
for f in t_16x65472 t_16x65473 t_16x65535; do for o in "--yuv 444 --negate" "--grayscale Average" "--grayscale Rec601Luma" "--lossless --negate" ; do try $B ~/adv/tall/$f.png $T/o.avif --preset turbo3 $o; done; done
echo "== PR batch: a tall frame in a list"
mkdir -p $T/b; rm -f $T/b/*; ls ~/adv/trim/bar_*_200x120.png | head -3 > $T/l; echo ~/adv/tall/t_16x65535.png >> $T/l; ls ~/adv/trim/bar_*_64x64.png | head -3 >> $T/l
for o in "--grayscale Average" "--yuv 444 --negate" "--lossless --negate"; do rm -rf $T/out; timeout 200 $B --batch $T/l --outdir $T/out --preset turbo3 $o > $T/o.txt 2> $T/e.txt; echo "rc=$? files=$(ls $T/out 2>/dev/null| wc -l) $o: $(grep -v 'stage ms' $T/e.txt | tail -2 | cut -c1-200)"; done
