#!/bin/bash
# identity_pal.sh [A B]: byte identity on the palette study's screen / flat_png / flat_jpg sets (~/pal/sets,
# prep_sets.py) at default, --palette on, --yuv 444 --depth 10 and --lossless -> ~/results/identity_pal.txt
A=${1:-~/old/build/av1gpu}; B=${2:-~/new/build/av1gpu}; TG=${3:+_$3}
R=~/results/identity_pal$TG.txt; : > $R
C="bash $HOME/cmpset.sh"
t0=$(date +%s)
{
echo "A=$A ($(cat $(dirname $(dirname $A))/COMMIT)) B=$B ($(cat $(dirname $(dirname $B))/COMMIT))"
for s in screen flat_png flat_jpg; do
  $C pal_${s}_default ~/pal/sets/$s $A $B
  $C pal_${s}_palette_on ~/pal/sets/$s $A $B --palette on
  $C pal_${s}_444_10 ~/pal/sets/$s $A $B --yuv 444 --depth 10
  $C pal_${s}_lossless ~/pal/sets/$s $A $B --lossless
done
echo "identity_pal took $(( $(date +%s) - t0 )) s"
} >> $R 2>&1
echo IDENTITY_PAL_DONE >> $R
