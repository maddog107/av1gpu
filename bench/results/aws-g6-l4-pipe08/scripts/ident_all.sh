#!/bin/bash
# ident_all.sh: byte identity of ~/new (this branch) against ~/old (main cde1f74), both sm_89 on this instance.
#  1. speed-v06's identity.sh (every set and option of the v0.6 release check, single-file mode included)
#  2. the palette sets (screen, flat_png, flat_jpg) at default, --palette on, --yuv 444 --depth 10, --lossless
#  3. short --batch lists (<= 8 inputs: one hardware queue since this branch): mixed.txt in 5 slices, default and
#     --yuv 444 --depth 10; the production pair (photo + its 512x384 thumbnail)
#  4. more single-file mode (this branch's main target): 70 files from every set, default; PNG / WebP also
#     --lossless; screen also --palette on; JPEGs also --yuv 444 --depth 10 and --preset slow
# Results: ~/results/identity.txt (one line per set), ~/results/ident_extra.txt.  Prints IDENT_ALL_DONE.
A=~/old/build/av1gpu; B=~/new/build/av1gpu
bash ~/identity.sh $A $B
R=~/results/ident_extra.txt; : > $R
C="bash $HOME/cmpset.sh"
{
for s in screen flat_png flat_jpg; do
  $C pal_${s}_default ~/pal/sets/$s $A $B
  $C pal_${s}_palette_on ~/pal/sets/$s $A $B --palette on
  $C pal_${s}_444_10 ~/pal/sets/$s $A $B --yuv 444 --depth 10
  $C pal_${s}_lossless ~/pal/sets/$s $A $B --lossless
done
mkdir -p ~/work/short; split -l 8 -d ~/sets/mixed.txt ~/work/short/mixed_
for l in ~/work/short/mixed_*; do
  $C short_$(basename $l) $l $A $B
  $C short_$(basename $l)_444_10 $l $A $B --yuv 444 --depth 10
done
printf "%s\n" ~/st/photo.jpg ~/st/thumb512.jpg > ~/work/short/pair.txt
$C short_pair ~/work/short/pair.txt $A $B
$C short_pair_444_10 ~/work/short/pair.txt $A $B --yuv 444 --depth 10
$C short_pair_slow ~/work/short/pair.txt $A $B --preset slow
# single-file mode
S=~/work/single_extra; rm -rf $S; mkdir -p $S
{ ls ~/sets/thumbs/* | awk 'NR % 28 == 1'; ls ~/images/*.jpg | awk 'NR % 10 == 5'; ls ~/images_std/*.jpg | awk 'NR % 10 == 2'
  ls ~/data/rgba/set/* | head -5; ls ~/sets/gray/* | head -5; ls ~/sets/odd/* | awk 'NR % 3 == 1'; ls ~/sets/webp/*
  ls ~/pal/sets/screen/* | head -6; ls ~/pal/sets/flat_jpg/* | head -4; ls ~/sets/mono/* | head -4; } > $S/list.txt
one() {   # name opts...: every file of list.txt that the options apply to (JPEGs skip --lossless)
  local n=$1; shift; local tot=0 d=0 fa=0
  mkdir -p $S/$n/a $S/$n/b
  while read f; do
    case "$f" in *.jpg|*.jpeg) [[ " $* " == *" --lossless "* ]] && continue;; esac
    tot=$((tot + 1)); b=$(basename $f)
    $A $f $S/$n/a/$b.avif "$@" > /dev/null 2>&1; ra=$?
    $B $f $S/$n/b/$b.avif "$@" > /dev/null 2>&1; rb=$?
    [ $ra = $rb ] || { fa=$((fa + 1)); echo "RC $n $b $ra/$rb"; }
    if [ -e $S/$n/a/$b.avif ] || [ -e $S/$n/b/$b.avif ]; then
      cmp -s $S/$n/a/$b.avif $S/$n/b/$b.avif || { d=$((d + 1)); echo "DIFF $n $b"; }
    fi
  done < $S/list.txt
  echo "single_$n: inputs $tot, files $(ls $S/$n/a | wc -l) / $(ls $S/$n/b | wc -l), differ $d, rc mismatches $fa, digest $(cd $S/$n/a && sha256sum * | sha256sum | cut -c1-16), opts [$*]"
}
one default
one lossless --lossless
one palette_on --palette on
one 444_10 --yuv 444 --depth 10
one slow --preset slow
one q90 --quality 90
# single-file --check / --debug-outputs still reconstruct (this branch turns the reconstruction off without them)
for f in ~/st/photo.jpg ~/st/t320.jpg ~/ll/kodak/kodim01.png; do
  $B $f /tmp/chk.avif --check 2>&1 | grep -E "check:" | sed "s#^#check $(basename $f): #"
  $B $f $S/dbg_b.avif --debug-outputs > /dev/null 2>&1; $A $f $S/dbg_a.avif --debug-outputs > /dev/null 2>&1
  for x in "" .obu .recon.yuv; do cmp -s $S/dbg_a.avif$x $S/dbg_b.avif$x && echo "debug-outputs $(basename $f) $x: same" || echo "DIFF debug-outputs $(basename $f) $x"; done
done
} >> $R 2>&1
echo IDENT_ALL_DONE >> $R
