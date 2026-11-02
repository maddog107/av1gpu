#!/bin/bash
# cmpset.sh NAME SRC BIN_A BIN_B [av1gpu options...]: encode SRC (a directory or a --batch LIST) with both binaries
# (the caller's environment, e.g. GP_ENCODERS, applies to both), cmp every output file.  One line per set:
# NAME: inputs, files A / B, files that differ or are missing, the skipped count, the digest of all of A's outputs.
N=$1; SRC=$2; A=$3; B=$4; shift 4
O=~/work/cmp/$N; rm -rf $O; mkdir -p $O
if [ -d "$SRC" ]; then nin=$(ls $SRC | wc -l); else nin=$(wc -l < $SRC); fi
ra=0; rb=0
$A --batch $SRC --outdir $O/a "$@" > $O/a.log 2>&1 || ra=$?
$B --batch $SRC --outdir $O/b "$@" > $O/b.log 2>&1 || rb=$?
na=$(ls $O/a 2>/dev/null | wc -l); nb=$(ls $O/b 2>/dev/null | wc -l); diff=0
for f in $O/a/*; do [ -e "$f" ] || continue; cmp -s $f $O/b/$(basename $f) || { diff=$((diff + 1)); echo "DIFF $N $(basename $f)"; }; done
for f in $O/b/*; do [ -e "$f" ] || continue; [ -e $O/a/$(basename $f) ] || { diff=$((diff + 1)); echo "ONLY_B $N $(basename $f)"; }; done
sk=$(grep -o 'skipped (unsupported): [0-9]*' $O/a.log | tail -1 | grep -o '[0-9]*$')
dig=$(cd $O/a 2>/dev/null && sha256sum * 2>/dev/null | sha256sum | cut -c1-16)
echo "$N: inputs $nin, files $na / $nb, differ $diff, skipped ${sk:-?}, rc $ra/$rb, digest $dig, opts [$*] ${GP_ENCODERS:+GP_ENCODERS=$GP_ENCODERS}"
[ $diff = 0 ] && [ $ra = 0 ] && [ $rb = 0 ] && rm -rf $O/a $O/b
true
