#!/bin/bash
# checks.sh: main (~/new) only.  The decoder gate (dav1d + aomdec, N/N), ctest, the GPU unit tests; results in
# ~/results/gate.txt and ~/results/ctest.txt.
B=~/new/build/av1gpu; G=~/new/tests/decoder_gate.py
mkdir -p ~/sets/gate30 ~/sets/gate_thumbs
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/sets/gate30/; done
ls ~/sets/thumbs/*.jpg | awk 'NR % 10 == 3' | while read f; do ln -sf $(readlink -f $f) ~/sets/gate_thumbs/; done
R=~/results/gate.txt; : > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 16 "$@" > ~/results/gate_$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate_$n.log >> $R; }
{
gate photos30_q60_120_200 --images ~/sets/gate30 --q 60,120,200
gate photos30_444_10_q120 --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
gate photos30_slow --images ~/sets/gate30 -- --preset slow
gate rgba --images ~/data/rgba/set --expect-alpha
gate kodak_lossless --images ~/ll/kodak --lossless
gate thumbs_q60_120 --images ~/sets/gate_thumbs --q 60,120
gate odd_q120 --images ~/sets/odd
gate odd_444_10 --images ~/sets/odd --expect-depth 10 -- --yuv 444 --depth 10
gate gray_q120 --images ~/sets/gray
# #39: low qindex (CDEF on with zero strengths; quality 95 / 90 are about qindex 13 / 26), slow, 4:4:4, mono
gate photos30_q13_26 --images ~/sets/gate30 --q 13,26
gate photos30_444_q13_26 --images ~/sets/gate30 --q 13,26 --expect-format 444 -- --yuv 444
gate mono_q13_26_120 --images ~/sets/mono --q 13,26,120
gate mono_slow --images ~/sets/mono -- --preset slow
gate photos30_444_slow --images ~/sets/gate30 --expect-format 444 -- --yuv 444 --preset slow
} 2>&1
echo GATE_DONE >> $R
C=~/results/ctest.txt; : > $C
{ cd ~/new/build && t=$(date +%s) && ctest --output-on-failure -j 4; echo "ctest rc $? ($(( $(date +%s) - t )) s)"
  ./gpu_fuzz_test --seed 1 --count 400; echo "gpu_fuzz_test rc $?"
  ./gpu_fuzz_test --seed 77 --count 200 --yuv444 60 --depth10 60 --lossless 30; echo "gpu_fuzz_test (444/10/ll) rc $?"
  ./gpu_postfilter_test --seed 1 --count 200; echo "gpu_postfilter_test rc $?"
  ./gpu_dsp_test; echo "gpu_dsp_test rc $?"
  ./varboost_test; echo "varboost_test rc $?"; } >> $C 2>&1
echo CTEST_DONE >> $C
