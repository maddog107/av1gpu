#!/bin/bash
# ident_fix.sh: the lossless sets were missing on the first pass (mk_ll_sets.py reads ~/br, a source tree: none on this
# instance), so ~/ll/{cmp,gray,misc,png16_*,graphics,jpeg,webp} were empty and the sets built from them (gray, webp,
# mixed.txt, sets/single/g0.png) incomplete.  This makes them (~/br -> ~/old, main's tree), rebuilds ~/sets and the mono
# set, and reruns every identity set that uses them, with the final branch (~/new = commits 1-6) against main; then
# ident_single.sh.  ~/results/identity_fix.txt; prints IDENT_FIX_DONE.
A=~/old/build/av1gpu; B=~/new/build/av1gpu
R=~/results/identity_fix.txt; : > $R
L=~/results/setup
ln -sfn ~/old ~/br
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets2.log 2>&1; echo "exit $?" >> $L/llsets2.log )
( cd ~ && python3 ~/mk_sets.py > $L/sets2.log 2>&1; echo "exit $?" >> $L/sets2.log )
( bash ~/mk_mono.sh > $L/mono2.log 2>&1; echo "exit $?" >> $L/mono2.log )
C="bash $HOME/cmpset.sh"
{
echo "A=$A (cde1f74) B=$B (final: commits 1-6); sets remade: $(tail -1 $L/llsets2.log) $(tail -1 $L/sets2.log) $(tail -1 $L/mono2.log)"
for d in ~/ll/*/ ~/sets/*/; do echo "  $d: $(ls $d | wc -l) files"; done; echo "  mixed.txt: $(wc -l < ~/sets/mixed.txt) lines"
$C mono_default ~/sets/mono $A $B
$C mono_q90 ~/sets/mono $A $B --quality 90
$C mono_q95 ~/sets/mono $A $B --quality 95
$C mono_slow ~/sets/mono $A $B --preset slow
$C mono_nosmall ~/sets/mono $A $B --small-preset none
$C mono_10 ~/sets/mono $A $B --depth 10
$C mono_tune_none ~/sets/mono $A $B --tune none
$C thumbs_default ~/sets/thumbs $A $B
$C gray_default ~/sets/gray $A $B
$C gray_lossless ~/sets/gray $A $B --lossless
$C misc_default ~/ll/misc $A $B
$C png16_10_default ~/ll/png16_10 $A $B --depth 10
$C png16_10_lossless ~/ll/png16_10 $A $B --lossless
$C llgraphics_lossless ~/ll/graphics $A $B --lossless --ycgco-re
$C webp_default ~/sets/webp $A $B
$C webp_lossless ~/sets/webp $A $B --lossless
$C mixed_list ~/sets/mixed.txt $A $B
$C mixed_list_444_10 ~/sets/mixed.txt $A $B --yuv 444 --depth 10
$C lljpeg_set ~/ll/jpeg $A $B
$C odd_default ~/sets/odd $A $B
$C kodak_default ~/ll/kodak $A $B
$C rgba_default ~/data/rgba/set $A $B
mkdir -p ~/work/short; rm -f ~/work/short/mixed_*; split -l 8 -d ~/sets/mixed.txt ~/work/short/mixed_
} >> $R 2>&1
echo IDENT_FIX_DONE >> $R
bash ~/ident_single.sh
