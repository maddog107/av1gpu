#!/bin/bash
# ident_single.sh: single-file mode identity of the final branch (~/new rebuilt with commits 5-6, which change only
# single-file mode) against main: identity.sh's single-file blocks (10 files x 3 option sets), ident_all.sh's 70-file
# single-file blocks, the --check / --debug-outputs checks, the short lists, and memcheck on single-file mode.
# ~/results/ident_single.txt; prints IDENT_SINGLE_DONE.
A=~/old/build/av1gpu; B=~/new/build/av1gpu
R=~/results/ident_single.txt; : > $R
{
echo "A=$A (cde1f74) B=$B (final: commits 1-6)"
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
S=~/work/single_extra2; rm -rf $S; mkdir -p $S
{ ls ~/sets/thumbs/* | awk 'NR % 28 == 1'; ls ~/images/*.jpg | awk 'NR % 10 == 5'; ls ~/images_std/*.jpg | awk 'NR % 10 == 2'
  ls ~/data/rgba/set/* | head -5; ls ~/sets/gray/*; ls ~/sets/odd/* | awk 'NR % 3 == 1'; ls ~/sets/webp/*
  ls ~/pal/sets/screen/* | head -6; ls ~/pal/sets/flat_jpg/* | head -4; ls ~/sets/mono/* | awk 'NR % 6 == 1'
  ls ~/ll/png16_10/* | head -3; ls ~/ll/misc/* | head -6; ls ~/ll/jpeg/* | awk 'NR % 3 == 1'; } > $S/list.txt
one() {
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
# error exits: same return code and message (unreadable file, a JPEG with --lossless, a crop outside the image)
for c in "/nonexistent.jpg" "$HOME/st/photo.jpg --lossless" "$HOME/st/t320.jpg --crop 999x999+0+0"; do
  $A $c /tmp/ea.avif > /tmp/ea.log 2>&1; ra=$?; $B $c /tmp/eb.avif > /tmp/eb.log 2>&1; rb=$?
  cmp -s /tmp/ea.log /tmp/eb.log && m=same || m=DIFFERENT; echo "error case [$c]: rc $ra/$rb, message $m"
done
for f in ~/st/photo.jpg ~/st/t320.jpg ~/ll/kodak/kodim01.png; do
  $B $f /tmp/chk.avif --check 2>&1 | grep -E "check:" | sed "s#^#check $(basename $f): #"
  $B $f $S/dbg_b.avif --debug-outputs > /dev/null 2>&1; $A $f $S/dbg_a.avif --debug-outputs > /dev/null 2>&1
  for x in "" .obu .recon.yuv; do cmp -s $S/dbg_a.avif$x $S/dbg_b.avif$x && echo "debug-outputs $(basename $f) $x: same" || echo "DIFF debug-outputs $(basename $f) $x"; done
done
$B ~/st/photo.jpg /tmp/st.avif --stats | head -c 300; echo
for l in ~/work/short/mixed_* ~/work/short/pair.txt; do bash ~/cmpset.sh final_short_$(basename $l) $l $A $B; done
CS=/usr/local/cuda/bin/compute-sanitizer
for f in ~/st/photo.jpg ~/st/t320.jpg ~/data/rgba/set/$(ls ~/data/rgba/set | head -1) $(ls ~/images/*_huge.jpg | head -1); do
  $CS --tool memcheck $B $f /tmp/mc.avif 2>&1 | tail -1 | sed "s#^#memcheck single $(basename $f): #"
done
} >> $R 2>&1
echo IDENT_SINGLE_DONE >> $R
