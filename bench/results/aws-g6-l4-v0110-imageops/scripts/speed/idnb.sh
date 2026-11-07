#!/bin/bash
# idnb.sh BASE TEST TAG: the bordered and all-blank pages (cuts really happen), every trim configuration, both formats; byte compare
A=$1; B=$2; TAG=$3; R=~/results/idnb_$TAG.txt; O=/dev/shm/idnb; : > $R; TOT=0; BAD=0
C444="--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100"
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
for src in ~/sets/border ~/sets/blank; do for fmt in "" "$C444"; do
  for cfg in "t0|--trim --fuzz 0%" "t10|--trim --fuzz 10%" "conv|$CONV" "m|--trim --fuzz 2% --trim-margin 1%" "auto|--auto-level" "tauto|--trim --fuzz 10% --auto-level" "crop|--crop 2001x2999+51+33"; do
    n=$(basename $src)_${fmt:+444}_${cfg%%|*}; rm -rf $O; mkdir -p $O
    $A --batch $src --outdir $O/a $fmt ${cfg#*|} > $O/a.log 2>&1; ra=$?
    $B --batch $src --outdir $O/b $fmt ${cfg#*|} > $O/b.log 2>&1; rb=$?
    t=$(ls $O/a | wc -l); d=0
    for f in $O/a/*.avif; do [ -e "$f" ] || continue; cmp -s $f $O/b/$(basename $f) || { d=$((d + 1)); echo "  DIFF $n $(basename $f)" >> $R; }; done
    [ "$t" = "$(ls $O/b | wc -l)" ] || d=$((d + 1)); [ $ra = $rb ] || d=$((d + 1))
    TOT=$((TOT + t)); BAD=$((BAD + d)); echo "batch $n: $t files, $d differ (exit $ra $rb; $(grep -h 'trim:' $O/a.log | cut -c10-))" >> $R
  done; done; done
echo "IDENTITY_DONE $TAG: $TOT files, $BAD differ" >> $R
