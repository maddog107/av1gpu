#!/bin/bash
# wall time of main vs branch av1gpu batches (alternating, 3 runs each, the minimum), photos and graphics
M=~/av1gpu/build/av1gpu; BR=~/br/build/av1gpu
O=~/pal/speed; rm -rf $O; mkdir -p $O
ls ~/pal/std100/*.jpg > $O/std100.txt
ls ~/pal/sets/photo/*.jpg > $O/sizes.txt
ls ~/pal/sets/photo/*.png > $O/kodak.txt
ls ~/pal/sets/screen/*.png ~/pal/sets/flat_png/*.png > $O/graphics.txt
t() {   # name list args...
  local n=$1 l=$2; shift 2
  local bm=999 bb=999
  for i in 1 2 3; do
    for b in main br; do
      bin=$M; [ $b = br ] && bin=$BR
      rm -rf $O/out; mkdir -p $O/out
      t0=$(date +%s.%N); $bin --batch $l --outdir $O/out "$@" > /dev/null 2>&1; t1=$(date +%s.%N)
      s=$(echo "$t1 - $t0" | bc)
      if [ $b = main ]; then bm=$(echo "if ($s < $bm) $s else $bm" | bc); else bb=$(echo "if ($s < $bb) $s else $bb" | bc); fi
    done
  done
  echo "SPEED $n main $bm s branch $bb s ratio $(echo "scale=3; $bb / $bm" | bc)"
}
t photos_std100 $O/std100.txt
t photos_sizes $O/sizes.txt
t photos_sizes_444_10 $O/sizes.txt --yuv 444 --depth 10
t kodak_lossless $O/kodak.txt --lossless
t graphics_default $O/graphics.txt
t graphics_444_10 $O/graphics.txt --yuv 444 --depth 10
t graphics_lossless $O/graphics.txt --lossless
t graphics_lossless_ycgco $O/graphics.txt --lossless --ycgco-re
