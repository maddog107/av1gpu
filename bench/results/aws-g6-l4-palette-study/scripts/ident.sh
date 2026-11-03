#!/bin/bash
# byte identity of the branch against main on photos: every output file cmp'd
#   the standard set (bench/openimages_v4_validation_100.csv) and the size range (bench/openimages_v4_gamut.csv):
#   default, --yuv 444 --depth 10
#   Kodak (24 PNG): --lossless, --lossless --ycgco-re
# usage: ident.sh [MAINBIN] [BRBIN]
M=${1:-~/av1gpu/build/av1gpu}; BR=${2:-~/br/build/av1gpu}
O=~/pal/ident; rm -rf $O; mkdir -p $O
ls ~/pal/std100/*.jpg > $O/std100.txt
ls ~/pal/sizerange/*.jpg > $O/sizes.txt   # the size range: bench/openimages_v4_gamut.csv (100 JPEGs)
ls ~/pal/sets/photo/*.png > $O/kodak.txt
run() {   # name list args...
  local n=$1 l=$2; shift 2
  for b in main br; do
    bin=$M; [ $b = br ] && bin=$BR
    mkdir -p $O/$n/$b
    t0=$(date +%s.%N)
    $bin --batch $l --outdir $O/$n/$b "$@" > $O/$n/$b.log 2>&1 || echo "$n $b exit $?"
    t1=$(date +%s.%N)
    echo "$n $b $(echo "$t1 - $t0" | bc) s"
  done
  local tot=0 diff=0
  for f in $O/$n/main/*.avif; do
    tot=$((tot + 1))
    cmp -s $f $O/$n/br/$(basename $f) || { diff=$((diff + 1)); echo "  DIFF $n $(basename $f)"; }
  done
  echo "IDENT $n: $((tot - diff))/$tot identical ($(ls $O/$n/br | wc -l) branch files)"
}
run std100_default $O/std100.txt
run std100_444_10 $O/std100.txt --yuv 444 --depth 10
run sizes_default $O/sizes.txt
run sizes_444_10 $O/sizes.txt --yuv 444 --depth 10
run kodak_ll $O/kodak.txt --lossless
run kodak_ll_ycgco $O/kodak.txt --lossless --ycgco-re
