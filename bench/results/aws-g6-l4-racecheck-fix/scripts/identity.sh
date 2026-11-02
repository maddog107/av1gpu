#!/bin/bash
# identity.sh: byte identity of the fix (B = ~/new, 78e374c) against main v0.6.0 (A = ~/old, d7d94b3) on the sets of
# the race078 brief; cmpset.sh compares every output file.  ~/results/identity.txt, IDENTITY_DONE at the end.
A=~/old/build/av1gpu; B=~/new/build/av1gpu
R=~/results/identity.txt; : > $R
C="bash $HOME/cmpset.sh"
t0=$(date +%s)
{
echo "A=$A ($(cat ~/old/COMMIT)) B=$B ($(cat ~/new/COMMIT))"
# size range (100 JPEGs)
$C sr_default ~/images $A $B
$C sr_turbo3_nosmall ~/images $A $B --preset turbo3 --small-preset none
$C sr_444 ~/images $A $B --yuv 444
$C sr_10 ~/images $A $B --depth 10
$C sr_444_10 ~/images $A $B --yuv 444 --depth 10
$C sr_slow_nosmall ~/images $A $B --preset slow --small-preset none
GP_ENCODERS=1 $C sr_default_enc1 ~/images $A $B
GP_ENCODERS=6 $C sr_default_enc6 ~/images $A $B
# standard set (100)
$C std_default ~/images_std $A $B
# Kodak (24 PNGs)
$C kodak_default ~/ll/kodak $A $B
$C kodak_lossless_ycgco ~/ll/kodak $A $B --lossless --ycgco-re
# RGBA (39 PNGs)
$C rgba_default ~/data/rgba/set $A $B
# thumbnails (mk_sets.py: 512x384 downscales, 320x320 thumbnails, 512x384 crops)
$C thumbs_default ~/sets/thumbs $A $B
echo "identity took $(( $(date +%s) - t0 )) s"
} >> $R 2>&1
echo IDENTITY_DONE >> $R
