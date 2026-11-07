#!/bin/bash
# idn65.sh BASE TEST TAG: TEST's outputs against BASE's, byte for byte, with the options that reach the changed code.
# Real sets once each, the adversarial set with every trim configuration, batch and single file.  Exit statuses compared too.
A=$1; B=$2; TAG=$3; R=~/results/idn_$TAG.txt; O=/dev/shm/idn; : > $R; TOT=0; BAD=0
C444="--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100"
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
batch() { # name src opts...
  local n=$1 src=$2; shift 2; rm -rf $O; mkdir -p $O
  $A --batch $src --outdir $O/a "$@" > $O/a.log 2>&1; local ra=$?
  $B --batch $src --outdir $O/b "$@" > $O/b.log 2>&1; local rb=$?
  local t=$(ls $O/a 2>/dev/null | wc -l) d=0 f
  for f in $O/a/*.avif; do [ -e "$f" ] || continue; cmp -s $f $O/b/$(basename $f) || { d=$((d + 1)); echo "  DIFF $n $(basename $f)" >> $R; }; done
  [ "$t" = "$(ls $O/b 2>/dev/null | wc -l)" ] || { d=$((d + 1)); echo "  COUNT $n $t $(ls $O/b | wc -l)" >> $R; }
  [ $ra = $rb ] || { d=$((d + 1)); echo "  EXIT $n $ra $rb" >> $R; }
  TOT=$((TOT + t)); BAD=$((BAD + d)); echo "batch $n: $t files, $d differ (exit $ra $rb; $(grep -h 'trim:' $O/a.log | cut -c10-))" >> $R
}
single() { # name list opts...
  local n=$1 l=$2; shift 2; local t=0 d=0 f; rm -rf $O; mkdir -p $O
  for f in $(cat $l); do
    $A $f $O/a.avif "$@" > /dev/null 2>&1; local ra=$?
    $B $f $O/b.avif "$@" > /dev/null 2>&1; local rb=$?
    t=$((t + 1))
    if [ $ra != $rb ] || { [ -e $O/a.avif ] && ! cmp -s $O/a.avif $O/b.avif; }; then d=$((d + 1)); echo "  DIFF $n $f ($ra $rb)" >> $R; fi
    rm -f $O/a.avif $O/b.avif
  done
  TOT=$((TOT + t)); BAD=$((BAD + d)); echo "single $n: $t files, $d differ" >> $R
}
for s in comics sr std thumbs pngwebp; do
  src=~/sets/$s; [ $s = sr ] && src=~/images; [ $s = std ] && src=~/images_std
  batch ${s}_trim0 $src --trim --fuzz 0%
  batch ${s}_trim10 $src --trim --fuzz 10%
  batch ${s}_conv $src $CONV
  batch ${s}_level $src --level 5%,95%,1.1
  batch ${s}_gamma $src --gamma 1.2
  batch ${s}_auto $src --auto-level
done
batch comics444_conv ~/sets/comics $C444 $CONV
batch comics444_auto ~/sets/comics $C444 --auto-level
batch comics444_level ~/sets/comics $C444 --level 5%,95%,1.1
# p5 changes the plain --crop path too (a crop without a resize): crops at even and odd offsets, 4:2:0 / 4:4:4 / gray JPEG,
# PNG (RGB, RGBA, gray, 16-bit), --depth 10, single file and batch
batch crop_odd_sr ~/images --crop 1001x777+33+17
batch crop_even_sr ~/images --crop 1000x776+32+16
batch crop_odd_d10 ~/images --crop 999x1001+7+9 --depth 10
batch crop_444_d10 ~/sets/comics --crop 1501x1203+101+77 --yuv 444 --depth 10
batch crop_pngwebp ~/sets/pngwebp --crop 641x479+3+5
batch crop_maxsize ~/images --crop 1201x901+11+13 --max-size 800x800
for seed in 1 2 3; do
  python3 ~/mk_trimfuzz.py $seed 900 >> $R
  F=~/sets/fuzz
  for cfg in "t0|--trim --fuzz 0%" "t3|--trim --fuzz 3%" "t10|--trim --fuzz 10%" "t40|--trim --fuzz 40%" "m|--trim --fuzz 5% --trim-margin 2" "cr|--crop 200x150+7+5 --trim --fuzz 2%" \
             "ll|--lossless --trim --fuzz 0%" "lr|--lossless --ycgco-re --trim --fuzz 2%" "d10|--depth 10 --yuv 444 --trim --fuzz 4%" "al|--auto-level" "lv|--level 3%,97%,0.8" "c1|--crop 37x29+3+5" "c2|--crop 64x64+2+2 --depth 10" "c3|--crop 100x100+1+0 --yuv 444 --depth 10"; do
    batch f${seed}_${cfg%%|*} $F ${cfg#*|} --max-mp 100 --max-input-mp 100
  done
  ls $F/* | awk 'NR % 15 == 0' > ~/work/fuzz1.txt
  single f${seed}_t0 ~/work/fuzz1.txt --trim --fuzz 0%
  single f${seed}_t10 ~/work/fuzz1.txt --trim --fuzz 10%
  single f${seed}_al ~/work/fuzz1.txt --auto-level
  single f${seed}_c1 ~/work/fuzz1.txt --crop 37x29+3+5
done
echo "IDENTITY_DONE $TAG: $TOT files, $BAD differ" >> $R
