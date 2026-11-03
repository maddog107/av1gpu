#!/bin/bash
# rerun8.sh: after identity8.sh, the sets that were empty or short in its run (fix_llsets.sh made them since): the
# monochrome set (mk_mono.sh again, now with ll/gray and ll/jpeg's gray inputs), gray, misc, png16_10, graphics, WebP,
# ll/jpeg, and single-file mode (br1).  ~/results/identity_rerun.txt
until grep -q IDENTITY_DONE ~/results/identity.txt && grep -q FIX_DONE ~/fix.out; do sleep 5; done
A=~/main/build/av1gpu; B=~/br/build/av1gpu
R=~/results/identity_rerun.txt; : > $R
C="bash $HOME/cmpset.sh"
bash ~/mk_mono.sh > ~/results/mono2.txt 2>&1
{
echo "A=$A (cde1f74) B=$B (br)"
$C mono_default ~/sets/mono $A $B
$C mono_q90 ~/sets/mono $A $B --quality 90
$C mono_q95 ~/sets/mono $A $B --quality 95
$C mono_slow ~/sets/mono $A $B --preset slow
$C mono_nosmall ~/sets/mono $A $B --small-preset none
$C mono_10 ~/sets/mono $A $B --depth 10
$C mono_tune_none ~/sets/mono $A $B --tune none
$C gray_default ~/sets/gray $A $B
$C gray_lossless ~/sets/gray $A $B --lossless
$C misc_default ~/ll/misc $A $B
$C png16_10_default ~/ll/png16_10 $A $B --depth 10
$C png16_10_lossless ~/ll/png16_10 $A $B --lossless
$C llgraphics_lossless ~/ll/graphics $A $B --lossless --ycgco-re
$C webp_default ~/sets/webp $A $B
$C webp_lossless ~/sets/webp $A $B --lossless
$C lljpeg_set ~/ll/jpeg $A $B
} >> $R 2>&1
echo IDENTITY_DONE >> $R
bash ~/identity_single.sh $A $B br1
