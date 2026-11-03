#!/bin/bash
# quick.sh TAG: quick identity (std / thumbs / size range default, std 4:4:4 10-bit) of ~/new against ~/old, then
# ncu kernel times of ~/new on std and thumbs -> ~/results/quick_TAG.txt
T=$1; R=~/results/quick_$T.txt; : > $R
A=~/old/build/av1gpu; B=~/new/build/av1gpu; C="bash $HOME/cmpset.sh"
$C q_std ~/images_std $A $B >> $R
$C q_thumbs ~/sets/thumbs $A $B >> $R
$C q_sr ~/images $A $B >> $R
$C q_std_444_10 ~/images_std $A $B --yuv 444 --depth 10 >> $R
$C q_sr_uvdp ~/images_std $A $B --uv-in-dp >> $R
SETS="std:$HOME/images_std thumbs:$HOME/sets/thumbs" bash ~/prof.sh $B $T > /dev/null 2>&1
for s in std thumbs; do echo "== $s"; grep -E "total|k_analysis |k_analysis_small|k_recon " ~/results/prof/${T}_${s}_ksum.txt | head -4; done >> $R
echo QUICK_DONE >> $R
