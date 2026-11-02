#!/bin/bash
# cmpsets.sh BIN_A BIN_B TAG [ENV...]: outputs of BIN_A and BIN_B, cmp of every file, on
#   sr       the 100 size-range JPEGs (~/images), defaults
#   sr444    the same, --yuv 444 --depth 10
#   std      the 100-photo standard set (~/images_std), defaults
# ENV (VAR=value ...) applies to both runs.  Outputs in ~/cmp/TAG/<set>/{a,b}.
A=$1; B=$2; T=$3; shift 3
W=~/cmp/$T; rm -rf $W; mkdir -p $W
for s in sr sr444 std; do
  case $s in sr) src=~/images; o=();; sr444) src=~/images; o=(--yuv 444 --depth 10);; std) src=~/images_std; o=();; esac
  for x in a b; do
    [ $x = a ] && bin=$A || bin=$B
    env "$@" $bin --batch $src --outdir $W/$s/$x "${o[@]}" > $W/$s.$x.log 2>&1 || echo "$s $x rc=$?"
  done
  n=$(ls $W/$s/a | wc -l); m=$(ls $W/$s/b | wc -l); d=0
  for f in $W/$s/a/*; do cmp -s $f $W/$s/b/$(basename $f) || d=$((d+1)); done
  echo "$T $s: $n vs $m files, $d differ ($(du -sk $W/$s/a | cut -f1) kB)"
done
