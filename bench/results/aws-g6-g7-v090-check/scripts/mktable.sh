#!/bin/bash
# mktable.sh: ~/sets9.tsv, the identity runs of the v0.9.0 release check, one per line:
#   kind|name|source|environment|av1gpu options|groups
# kind batch = `av1gpu --batch SOURCE --outdir ...`, single = `av1gpu FILE OUT` for every file of the list SOURCE
# (JPEGs skipped with --lossless), err = one single-file run that must fail the same way.
# groups: full = every run (speed-v06's identity.sh list, the palette sets, short --batch lists, single-file mode);
#         sub  = the CUDA 12 subset (B12 against A12 and B13).
T=~/sets9.tsv; : > $T
a() { echo "$1|$2|$3|$4|$5|$6" >> $T; }
I=~/images; S=~/images_std
# speed-v06 identity.sh, in its order
a batch sr_default $I "" "" "full sub"
a batch sr_turbo3_nosmall $I "" "--preset turbo3 --small-preset none" "full"
a batch sr_nosmall $I "" "--small-preset none" "full"
a batch sr_444 $I "" "--yuv 444" "full"
a batch sr_10 $I "" "--depth 10" "full"
a batch sr_444_10 $I "" "--yuv 444 --depth 10" "full sub"
a batch sr_base_nosmall $I "" "--preset base --small-preset none" "full"
a batch sr_slow_nosmall $I "" "--preset slow --small-preset none" "full"
a batch sr_fast_nosmall $I "" "--preset fast --small-preset none" "full"
a batch sr_q30 $I "" "--quality 30" "full"
a batch sr_q90 $I "" "--quality 90" "full"
a batch sr_q95 $I "" "--quality 95" "full"
a batch sr_tune_none $I "" "--tune none" "full"
a batch sr_tiles_2_1_1 $I "" "--tiles 2,1,1" "full"
a batch sr_default_enc1 $I "GP_ENCODERS=1" "" "full"
a batch sr_default_enc6 $I "GP_ENCODERS=6" "" "full"
a batch sr_slow $I "" "--preset slow" "full"
a batch sr_444_q90 $I "" "--yuv 444 --quality 90" "full"
a batch sr_444_q95 $I "" "--yuv 444 --quality 95" "full"
a batch sr_444_slow $I "" "--yuv 444 --preset slow" "full"
a batch sr_444_10_q95 $I "" "--yuv 444 --depth 10 --quality 95" "full"
a batch sr_10_q95 $I "" "--depth 10 --quality 95" "full"
for o in "default|" "q90|--quality 90" "q95|--quality 95" "slow|--preset slow" "nosmall|--small-preset none" "10|--depth 10" "tune_none|--tune none"; do
  a batch mono_${o%%|*} ~/sets/mono "" "${o#*|}" "full"; done
a batch std_q95 $S "" "--quality 95" "full"
a batch std_slow $S "" "--preset slow" "full"
a batch thumbs_q95 ~/sets/thumbs "" "--quality 95" "full"
a batch thumbs_slow ~/sets/thumbs "" "--preset slow" "full"
a batch kodak_q95 ~/ll/kodak "" "--quality 95" "full"
a batch kodak_444_q90 ~/ll/kodak "" "--yuv 444 --quality 90" "full"
a batch rgba_q95 ~/data/rgba/set "" "--quality 95" "full"
a batch odd_q95 ~/sets/odd "" "--quality 95" "full"
a batch odd_slow ~/sets/odd "" "--preset slow" "full"
a batch std_default $S "" "" "full sub"
a batch std_nosmall $S "" "--small-preset none" "full"
a batch thumbs_default ~/sets/thumbs "" "" "full sub"
a batch thumbs_nosmall ~/sets/thumbs "" "--small-preset none" "full"
a batch kodak_default ~/ll/kodak "" "" "full"
a batch kodak_lossless ~/ll/kodak "" "--lossless" "full sub"
a batch kodak_lossless_ycgco ~/ll/kodak "" "--lossless --ycgco-re" "full"
a batch rgba_default ~/data/rgba/set "" "" "full sub"
a batch rgba_aq100 ~/data/rgba/set "" "--alpha-quality 100" "full"
a batch rgba_lossless ~/data/rgba/set "" "--lossless" "full sub"
a batch rgba_no_alpha ~/data/rgba/set "" "--no-alpha" "full"
a batch gray_default ~/sets/gray "" "" "full"
a batch gray_lossless ~/sets/gray "" "--lossless" "full"
a batch odd_default ~/sets/odd "" "" "full"
a batch odd_444_10 ~/sets/odd "" "--yuv 444 --depth 10" "full"
a batch odd_lossless ~/sets/odd "" "--lossless" "full"
a batch misc_default ~/ll/misc "" "" "full"
a batch png16_10_default ~/ll/png16_10 "" "--depth 10" "full"
a batch png16_10_lossless ~/ll/png16_10 "" "--lossless" "full"
a batch llgraphics_lossless ~/ll/graphics "" "--lossless --ycgco-re" "full sub"
a batch webp_default ~/sets/webp "" "" "full"
a batch webp_lossless ~/sets/webp "" "--lossless" "full"
a batch mixed_list ~/sets/mixed.txt "" "" "full"
a batch mixed_list_444_10 ~/sets/mixed.txt "" "--yuv 444 --depth 10" "full"
a batch lljpeg_set ~/ll/jpeg "" "" "full"
# the palette sets (v0.8.0: --palette auto by default)
for s in screen flat_png flat_jpg; do
  a batch pal_${s}_default ~/pal/sets/$s "" "" "full sub"
  a batch pal_${s}_palette_on ~/pal/sets/$s "" "--palette on" "full sub"
  a batch pal_${s}_444_10 ~/pal/sets/$s "" "--yuv 444 --depth 10" "full sub"
  a batch pal_${s}_lossless ~/pal/sets/$s "" "--lossless" "full sub"
done
# short --batch lists (<= 8 encodes: one hardware queue in v0.9.0) and the production pair (photo + 512 x 384 thumbnail)
for l in ~/lists/short/mixed_*; do
  a batch short_$(basename $l) $l "" "" "full"
  a batch short_$(basename $l)_444_10 $l "" "--yuv 444 --depth 10" "full"
done
a batch short_pair ~/lists/short/pair.txt "" "" "full sub"
a batch short_pair_444_10 ~/lists/short/pair.txt "" "--yuv 444 --depth 10" "full sub"
a batch short_pair_slow ~/lists/short/pair.txt "" "--preset slow" "full"
a batch short_pair_q90 ~/lists/short/pair.txt "" "--quality 90" "full"
# single-file mode: identity.sh's 10 files x 3, and every set (~/lists/single_all.txt)
a single single10_default ~/lists/single10.txt "" "" "full sub"
a single single10_444_10 ~/lists/single10.txt "" "--yuv 444 --depth 10" "full sub"
a single single10_slow ~/lists/single10.txt "" "--preset slow" "full"
a single single_all_default ~/lists/single_all.txt "" "" "full sub"
a single single_all_lossless ~/lists/single_all.txt "" "--lossless" "full sub"
a single single_all_palette_on ~/lists/single_all.txt "" "--palette on" "full"
a single single_all_444_10 ~/lists/single_all.txt "" "--yuv 444 --depth 10" "full"
a single single_all_slow ~/lists/single_all.txt "" "--preset slow" "full"
a single single_all_q90 ~/lists/single_all.txt "" "--quality 90" "full"
a single single_st_default ~/lists/single_st.txt "" "" "full sub"
a single single_st_444_10 ~/lists/single_st.txt "" "--yuv 444 --depth 10" "full"
# error exits in single-file mode: same return code and message
a err err_missing /nonexistent.jpg "" "" "full"
a err err_jpeg_lossless ~/st/photo.jpg "" "--lossless" "full"
a err err_crop ~/st/t320.jpg "" "--crop 999x999+2000+2000" "full"
ls ~/st/photo.jpg ~/st/big.jpg ~/st/thumb512.jpg ~/st/t320.jpg > ~/lists/single_st.txt
echo "$(wc -l < $T) runs: $(grep -c ' full' $T) full, $(grep -c ' sub' $T) sub"
