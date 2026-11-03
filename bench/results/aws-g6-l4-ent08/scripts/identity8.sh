#!/bin/bash
# identity8.sh [A B [TAG]]: byte identity of the branch (B, default ~/br) against main cde1f74 (A, default ~/main) on
# every set and option of speed-v06's identity.sh, plus the palette sets of the palette study (screen, flat_png,
# flat_jpg; prep_sets.py) at the default, --palette on, --yuv 444 --depth 10 and --lossless;
# ~/results/identity.txt (one line per set, DIFF lines for any difference).
A=${1:-~/main/build/av1gpu}; B=${2:-~/br/build/av1gpu}
R=~/results/identity${3:+_$3}.txt; : > $R
C="bash $HOME/cmpset.sh"
t0=$(date +%s)
{
echo "A=$A ($(cat $(dirname $(dirname $A))/COMMIT)) B=$B ($(cat $(dirname $(dirname $B))/COMMIT))"
# size range (100 JPEGs)
$C sr_default ~/images $A $B
$C sr_turbo3_nosmall ~/images $A $B --preset turbo3 --small-preset none
$C sr_nosmall ~/images $A $B --small-preset none
$C sr_444 ~/images $A $B --yuv 444
$C sr_10 ~/images $A $B --depth 10
$C sr_444_10 ~/images $A $B --yuv 444 --depth 10
$C sr_base_nosmall ~/images $A $B --preset base --small-preset none
$C sr_slow_nosmall ~/images $A $B --preset slow --small-preset none
$C sr_fast_nosmall ~/images $A $B --preset fast --small-preset none
$C sr_q30 ~/images $A $B --quality 30
$C sr_q90 ~/images $A $B --quality 90
$C sr_q95 ~/images $A $B --quality 95
$C sr_tune_none ~/images $A $B --tune none
$C sr_tiles_2_1_1 ~/images $A $B --tiles 2,1,1
GP_ENCODERS=1 $C sr_default_enc1 ~/images $A $B
GP_ENCODERS=6 $C sr_default_enc6 ~/images $A $B
# #39 (CDEF apply reuses the search's maps and directions): slow with the small preset, q90 / q95 (CDEF on with zero
# strengths), 4:4:4 and monochrome
$C sr_slow ~/images $A $B --preset slow
$C sr_444_q90 ~/images $A $B --yuv 444 --quality 90
$C sr_444_q95 ~/images $A $B --yuv 444 --quality 95
$C sr_444_slow ~/images $A $B --yuv 444 --preset slow
$C sr_444_10_q95 ~/images $A $B --yuv 444 --depth 10 --quality 95
$C sr_10_q95 ~/images $A $B --depth 10 --quality 95
$C mono_default ~/sets/mono $A $B
$C mono_q90 ~/sets/mono $A $B --quality 90
$C mono_q95 ~/sets/mono $A $B --quality 95
$C mono_slow ~/sets/mono $A $B --preset slow
$C mono_nosmall ~/sets/mono $A $B --small-preset none
$C mono_10 ~/sets/mono $A $B --depth 10
$C mono_tune_none ~/sets/mono $A $B --tune none
$C std_q95 ~/images_std $A $B --quality 95
$C std_slow ~/images_std $A $B --preset slow
$C thumbs_q95 ~/sets/thumbs $A $B --quality 95
$C thumbs_slow ~/sets/thumbs $A $B --preset slow
$C kodak_q95 ~/ll/kodak $A $B --quality 95
$C kodak_444_q90 ~/ll/kodak $A $B --yuv 444 --quality 90
$C rgba_q95 ~/data/rgba/set $A $B --quality 95
$C odd_q95 ~/sets/odd $A $B --quality 95
$C odd_slow ~/sets/odd $A $B --preset slow
# standard set (100)
$C std_default ~/images_std $A $B
$C std_nosmall ~/images_std $A $B --small-preset none
# thumbnails / small images
$C thumbs_default ~/sets/thumbs $A $B
$C thumbs_nosmall ~/sets/thumbs $A $B --small-preset none
# PNG and WebP
$C kodak_default ~/ll/kodak $A $B
$C kodak_lossless ~/ll/kodak $A $B --lossless
$C kodak_lossless_ycgco ~/ll/kodak $A $B --lossless --ycgco-re
$C rgba_default ~/data/rgba/set $A $B
$C rgba_aq100 ~/data/rgba/set $A $B --alpha-quality 100
$C rgba_lossless ~/data/rgba/set $A $B --lossless
$C rgba_no_alpha ~/data/rgba/set $A $B --no-alpha
$C gray_default ~/sets/gray $A $B
$C gray_lossless ~/sets/gray $A $B --lossless
$C odd_default ~/sets/odd $A $B
$C odd_444_10 ~/sets/odd $A $B --yuv 444 --depth 10
$C odd_lossless ~/sets/odd $A $B --lossless
$C misc_default ~/ll/misc $A $B
$C png16_10_default ~/ll/png16_10 $A $B --depth 10
$C png16_10_lossless ~/ll/png16_10 $A $B --lossless
$C llgraphics_lossless ~/ll/graphics $A $B --lossless --ycgco-re
$C webp_default ~/sets/webp $A $B
$C webp_lossless ~/sets/webp $A $B --lossless
$C mixed_list ~/sets/mixed.txt $A $B
$C mixed_list_444_10 ~/sets/mixed.txt $A $B --yuv 444 --depth 10
$C lljpeg_set ~/ll/jpeg $A $B
# palette sets (v0.8.0: palette symbols are coded by the GPU entropy coder)
for s in screen flat_png flat_jpg; do
  $C pal_${s}_default ~/pal/sets/$s $A $B
  $C pal_${s}_on ~/pal/sets/$s $A $B --palette on
  $C pal_${s}_444_10 ~/pal/sets/$s $A $B --yuv 444 --depth 10
  $C pal_${s}_lossless ~/pal/sets/$s $A $B --lossless
done
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
echo "identity took $(( $(date +%s) - t0 )) s"
} >> $R 2>&1
echo IDENTITY_DONE >> $R
