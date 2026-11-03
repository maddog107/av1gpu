#!/bin/bash
# switch3.sh: stop verify2.sh / after2.sh (change 2 without the __syncwarp before warp_residual's writes: racecheck
# flagged coeff_bits reads of the levels in res against the next residual's writes), keep the evidence, put the fixed
# k_analysis.cuh into ~/new2 and start verify2.sh + after2.sh again
for p in verify2.sh after2.sh checks2.sh identity.sh identity_pal.sh cmpset.sh sanitizer.sh race_ab.sh decoder_gate.py spd_an.py; do
  pkill -f "$p"; done
pkill -f compute-sanitizer; pkill -f "build/av1gpu"
sleep 3
pkill -9 -f compute-sanitizer; pkill -9 -f "build/av1gpu"
mkdir -p ~/results/v2pre
cp ~/results/identity_new2.txt ~/results/v2pre/identity_new2_partial.txt
cp ~/results/race/new2_0.log ~/results/v2pre/race_new2_0_partial.log
cp ~/results/sanitizer_new2.txt ~/results/v2pre/ 2>/dev/null
cp ~/results/gate_new2.txt ~/results/v2pre/ 2>/dev/null
cp ~/up3/k_analysis.cuh ~/new2/encoder/src/
rm -f ~/results/speed2.log ~/results/identity_rerun_new2.txt
nohup bash ~/verify2.sh > ~/verify2.out 2>&1 < /dev/null &
nohup bash ~/after2.sh > ~/after2.out 2>&1 < /dev/null &
echo switched
