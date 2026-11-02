#!/bin/bash
# ident.sh: byte identity of the branch (B = ~/cr) against main d7d94b3 (A = ~/main) without the new options, every set
# of the brief (and a few more), with cmpset.sh; then single-file mode; then the new options at values that change
# nothing (a box larger than every image, a crop of at least the whole image, --max-mp 35.6) against main's plain run.
# ~/results/identity.txt (one line per set, DIFF lines for any difference).
A=~/main/build/av1gpu; B=~/cr/build/av1gpu
R=~/results/identity.txt; : > $R
C="bash $HOME/cmpset.sh"
t0=$(date +%s)
{
echo "A=$A ($(cat ~/main/COMMIT)) B=$B ($(cat ~/cr/COMMIT))"
$C sr_default ~/images $A $B
$C sr_turbo3_nosmall ~/images $A $B --preset turbo3 --small-preset none
$C sr_444_10 ~/images $A $B --yuv 444 --depth 10
$C sr_10 ~/images $A $B --depth 10
$C std_default ~/images_std $A $B
$C kodak_default ~/ll/kodak $A $B
$C kodak_lossless_ycgco ~/ll/kodak $A $B --lossless --ycgco-re
$C kodak_lossless ~/ll/kodak $A $B --lossless
$C rgba_default ~/data/rgba/set $A $B
$C rgba_lossless ~/data/rgba/set $A $B --lossless
$C thumbs_default ~/sets/thumbs $A $B
$C odd_default ~/sets/odd $A $B
$C odd_444_10 ~/sets/odd $A $B --yuv 444 --depth 10
$C gray_default ~/sets/gray $A $B
$C webp_default ~/sets/webp $A $B
$C png16_10_default ~/ll/png16_10 $A $B --depth 10
$C mixed_list ~/sets/mixed.txt $A $B
$C lljpeg_set ~/ll/jpeg $A $B
# single-file mode (no --batch): 10 files one at a time
for opts in "" "--yuv 444 --depth 10" "--preset slow"; do
  O=~/work/single; rm -rf $O; mkdir -p $O/a $O/b; n=0; d=0; fa=0
  for f in ~/sets/single/*; do
    b=$(basename $f); n=$((n + 1))
    $A $f $O/a/$b.avif $opts > $O/a.log 2>&1 || fa=$((fa + 1))
    $B $f $O/b/$b.avif $opts > $O/b.log 2>&1 || fa=$((fa + 1))
    cmp -s $O/a/$b.avif $O/b/$b.avif || { d=$((d + 1)); echo "DIFF single $b [$opts]"; }
  done
  echo "single_file: inputs $n, files $(ls $O/a | wc -l) / $(ls $O/b | wc -l), differ $d, failed runs $fa, digest $(cd $O/a && sha256sum * | sha256sum | cut -c1-16), opts [$opts]"
done
# the new options at no-op values: main without them (A) against the branch with them (B)
noop() { local n=$1 src=$2; shift 2
  local O=~/work/noop/$n; rm -rf $O; mkdir -p $O
  $A --batch $src --outdir $O/a > $O/a.log 2>&1; local ra=$?
  $B --batch $src --outdir $O/b "$@" > $O/b.log 2>&1; local rb=$?
  local d=0; for f in $O/a/*; do cmp -s $f $O/b/$(basename $f) || d=$((d + 1)); done
  echo "noop_$n: files $(ls $O/a | wc -l) / $(ls $O/b | wc -l), differ $d, rc $ra/$rb, opts [$*]"; [ $d = 0 ] && rm -rf $O/a $O/b; }
noop sr_box ~/images --max-size 65536x65536
noop sr_crop ~/images --crop 65536x65536+0+0
noop sr_maxmp ~/images --max-mp 35.6 --max-input-mp 35.6
noop rgba_box ~/data/rgba/set --max-size 65536x65536 --crop 0,0,65536,65536
echo "identity took $(( $(date +%s) - t0 )) s"
} >> $R 2>&1
echo IDENTITY_DONE >> $R
