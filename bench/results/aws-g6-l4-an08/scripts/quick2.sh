#!/bin/bash
# quick2.sh: quick identity of ~/new2 against ~/old (std, thumbs, size range default, size range base preset,
# std 4:4:4 10-bit, size range 10-bit) -> ~/results/quick2.txt; then ncu occupancy of the 32x32 / 64x64 kernels
R=~/results/quick2.txt; : > $R
A=~/old/build/av1gpu; B=~/new2/build/av1gpu; C="bash $HOME/cmpset.sh"
$C q2_std ~/images_std $A $B >> $R
$C q2_thumbs ~/sets/thumbs $A $B >> $R
$C q2_sr ~/images $A $B >> $R
$C q2_sr_base ~/images $A $B --preset base --small-preset none >> $R
$C q2_std_444_10 ~/images_std $A $B --yuv 444 --depth 10 >> $R
$C q2_sr_10 ~/images $A $B --depth 10 >> $R
bash ~/ncu_detail.sh $B n2_3232 ~/images_std "k_analysis<.int.32, .int.32," 1
bash ~/ncu_detail.sh $B n2_6464 ~/images_std "k_analysis<.int.64, .int.64," 1
for k in n2_3232 n2_6464; do echo "== $k"; grep -E "Duration|Block Limit Shared|Theoretical Occupancy|Achieved Occupancy|Static Shared" ~/results/prof/${k}_detail.txt; done >> $R
echo QUICK2_DONE >> $R
