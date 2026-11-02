#!/bin/bash
# lossless bytes of the 23 graphics (screen_synth_*): main (study) vs branch (lltest), GBR and YCgCo-R
for m in gbr ycgco; do
  d=screen; [ $m = ycgco ] && d="screen--ycgco-re"
  a=0; b=0
  for f in ~/pal/out/gpu_ll/$m/screen/screen_synth_*.avif; do
    a=$((a + $(stat -c %s $f))); b=$((b + $(stat -c %s ~/pal/lltest/$d/auto/$(basename $f))))
  done
  echo "graphics $m main $a branch $b ratio $(echo "scale=4; $b / $a" | bc)"
  a=0; b=0
  for f in ~/pal/out/gpu_ll/$m/flat_png/*.avif; do
    [ -f ~/pal/lltest/flat_png/auto/$(basename $f) ] || continue
    a=$((a + $(stat -c %s $f))); b=$((b + $(stat -c %s ~/pal/lltest/flat_png/auto/$(basename $f))))
  done
  [ $m = gbr ] && echo "flat_png $m main $a branch $b ratio $(echo "scale=4; $b / $a" | bc)"
done
