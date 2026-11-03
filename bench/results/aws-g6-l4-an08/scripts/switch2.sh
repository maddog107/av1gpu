#!/bin/bash
# switch2.sh: stop verify.sh (change 1 only) and quick2.sh, keep their partial results, start verify2.sh (changes 1 + 2)
# and the ncu occupancy check of the new 32x32 / 64x64 / 64x32 kernel layouts
for p in verify.sh checks_an.sh quick2.sh identity.sh identity_pal.sh cmpset.sh sanitizer.sh race_ab.sh decoder_gate.py spd_an.py; do
  pkill -f "$p"; done
pkill -f compute-sanitizer; pkill -f "build/av1gpu"
sleep 3
pkill -9 -f compute-sanitizer; pkill -9 -f "build/av1gpu"
mkdir -p ~/results/v1
cp ~/results/identity.txt ~/results/v1/identity_new_partial.txt 2>/dev/null
cp ~/results/gate.txt ~/results/ctest.txt ~/results/v1/ 2>/dev/null
cp ~/results/sanitizer.txt ~/results/v1/sanitizer_new_partial.txt 2>/dev/null
cp ~/results/quick2.txt ~/results/v1/ 2>/dev/null
nohup bash ~/verify2.sh > ~/verify2.out 2>&1 < /dev/null &
( bash ~/ncu_detail.sh ~/new2/build/av1gpu n2_3232 ~/images_std "k_analysis<.int.32, .int.32," 1
  bash ~/ncu_detail.sh ~/new2/build/av1gpu n2_6464 ~/images_std "k_analysis<.int.64, .int.64," 1
  bash ~/ncu_detail.sh ~/new2/build/av1gpu n2_6432 ~/images_std "k_analysis<.int.64, .int.32," 1
  bash ~/ncu_detail.sh ~/old/build/av1gpu o_3232 ~/images_std "k_analysis<.int.32, .int.32," 1
  for k in n2_3232 o_3232 n2_6464 n2_6432; do echo "== $k"; grep -E "Duration|Block Limit Shared|Theoretical Occupancy|Achieved Occupancy|Static Shared" ~/results/prof/${k}_detail.txt; done > ~/results/occ2.txt
  echo OCC_DONE >> ~/results/occ2.txt ) > /dev/null 2>&1 < /dev/null &
ps aux | grep -E "verify|identity|sanitizer|av1gpu" | grep -v grep | wc -l
