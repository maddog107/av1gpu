#!/bin/bash
B=~/pr/build/av1gpu; M=~/main/build/av1gpu; T=/dev/shm/p2; mkdir -p $T
try() { rm -f $T/o.avif; s=$(date +%s.%N); timeout 200 "$@" > $T/out.txt 2> $T/err.txt; rc=$?; e=$(date +%s.%N)
  sz=""; if [ -f $T/o.avif ]; then sz=$(python3 - <<PY
import sys; sys.path.insert(0,'/home/ubuntu/t'); import avt
import numpy as np
try:
    c=avt.check('$T/o.avif'); y=c['planes'][0]; print(c['si'][:5], 'ok Y',int(y.min()),int(y.max()))
except Exception as ex: print('BAD', ex)
PY
); fi
  printf 'rc=%s %.1fs [%s] %s | %s\n' $rc $(echo "$e-$s"|bc) "$sz" "${*:1}" "$(tail -1 $T/err.txt | cut -c1-150)" | sed "s#$T/o.avif##; s#/home/ubuntu/##g"
}
echo "== tall on main"
for f in t_1x65535 t_65535x1 t_16x65535 t_65535x16 t_3x40000 t_8x16385 t_1x16384; do try $M ~/adv/tall/$f.png $T/o.avif --preset turbo3; done
echo "== tall on PR, with tone/gray/trim"
for f in t_1x65535 t_65535x1 t_16x65535 t_65535x16 t_3x40000 t_40000x3 t_8x16385 t_1x16384; do
 for o in "--gamma 2" "--negate" "--grayscale Rec709Luma" "--trim" "--auto-level" "--yuv 444 --negate" "--yuv 444 --grayscale Average" "--depth 10 --negate" "--lossless --negate" "--lossless --grayscale RMS"; do
  try $B ~/adv/tall/$f.png $T/o.avif --preset turbo3 $o
 done
done
