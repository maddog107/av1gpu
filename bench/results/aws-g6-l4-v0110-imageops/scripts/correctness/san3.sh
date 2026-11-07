#!/bin/bash
# memcheck / racecheck / synccheck on the fix head's 4:2:0 tone, gray and trim at awkward sizes, both frame kinds
CS=/usr/local/cuda/bin/compute-sanitizer; F=${BIN:-$HOME/fix/build/av1gpu}; R=${SANDIR:-$HOME/results/san3}; mkdir -p $R; D=/dev/shm/san3; rm -rf $D; mkdir -p $D/png $D/jpg $D/mix
python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, '/home/ubuntu/t')
from gen_lib import wpng
from PIL import Image
D='/dev/shm/san3'; rng=np.random.default_rng(9)
sizes=[(1,1),(1,2),(2,1),(2,2),(3,3),(1,37),(37,1),(1,64),(64,1),(2,33),(33,2),(5,7),(7,5),(63,65),(65,63),(127,129),(64,64),(65,65),(129,1),(1,129)]
for w,h in sizes:
    a=rng.integers(0,256,(h,w,3),dtype=np.uint8); a[:h//2]=255
    wpng(f'{D}/png/r{w}x{h}.png',a); Image.fromarray(a).save(f'{D}/jpg/r{w}x{h}.jpg',quality=90,subsampling=2)
    wpng(f'{D}/mix/r{w}x{h}.png',a); Image.fromarray(a).save(f'{D}/mix/j{w}x{h}.jpg',quality=90,subsampling=2)
    al=np.full((h,w),255,np.uint8); al[:,:max(1,w//3)]=40; wpng(f'{D}/mix/a{w}x{h}.png',np.dstack([a,al]))
    wpng(f'{D}/mix/d{w}x{h}.png',a.astype(np.uint16)*257+3,bits=16)
PY
n=$(ls $D/mix | wc -l); echo "$n mixed inputs"
run() { local tool=$1 tag=$2; shift 2; rm -rf $D/o; mkdir -p $D/o
  timeout 1500 $CS --tool $tool --error-exitcode 99 --print-limit 20 "$@" > $R/$tool.$tag.txt 2>&1; rc=$?
  echo "$tool $tag rc=$rc files=$(ls $D/o | wc -l): $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $R/$tool.$tag.txt | tail -1)"; }
OPTS=("t1|--negate" "t2|--level 10%,90%,1.3" "t3|--gamma 0.7 --auto-level" "t4|--grayscale Rec709Luma" "t5|--grayscale Lightness --resize 150%" "t6|--trim --fuzz 5%" "t7|--depth 10 --level 5%,95% --negate" "t8|--depth 10 --trim --auto-level --grayscale Average" "t9|--trim --fuzz 3% --gamma 2 --resize 77x77^")
CROPS=("g1|--crop 40x40+2+2 --grayscale Average" "g2|--crop 41x39+3+1 --grayscale Lightness" "g3|--crop 64x63+0+0 --colorspace Gray" "g4|--crop 20x20+44+44 --grayscale RMS" "c1|--crop 40x40+0+0" "c2|--crop 40x40+25+0" "c3|--crop 40x40+0+25" "c4|--crop 40x40+25+25" "c5|--crop 63x1+1+64" "c6|--crop 1x63+64+1" "c7|--crop 3x3+62+62")
for tool in memcheck racecheck synccheck; do
  for o in "${OPTS[@]}"; do tag=${o%%|*}; a=${o#*|}
    run $tool b_mix_$tag $F --batch $D/mix --outdir $D/o --small-preset none --preset turbo3 $a
    run $tool b_jpg_$tag $F --batch $D/jpg --outdir $D/o --small-preset none --preset turbo3 $a
    run $tool b_png_$tag $F --batch $D/png --outdir $D/o --small-preset none --preset turbo3 $a
  done
  for c in "${CROPS[@]}"; do tag=${c%%|*}; a=${c#*|}
    [ $tool = synccheck ] && continue
    run $tool cropj_$tag $F --batch $D/jpg --outdir $D/o --small-preset none --preset turbo3 $a
    run $tool crop_$tag $F --batch $D/mix --outdir $D/o --small-preset none --preset turbo3 $a --negate --gamma 1.2 --trim
    run $tool cropd_$tag $F --batch $D/mix --outdir $D/o --small-preset none --preset turbo3 --depth 10 $a --auto-level --grayscale RMS
  done
  [ $tool = memcheck ] || continue
  for f in r1x1 r1x37 r37x1 r2x2 r3x3 r65x63 r127x129; do for k in png jpg; do
    run $tool s_${k}_$f $F $D/$k/$f.$k $D/o/s.avif --negate --level 5%,95% --trim --fuzz 2%
    run $tool s10_${k}_$f $F $D/$k/$f.$k $D/o/s.avif --depth 10 --gamma 1.5 --grayscale Average
  done; done
done
echo SAN3_DONE
