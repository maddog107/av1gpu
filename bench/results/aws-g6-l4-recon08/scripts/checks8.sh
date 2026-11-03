#!/bin/bash
# checks8.sh: ~/new only.  Decoder gate (dav1d + aomdec) on photos, thumbnails, odd sizes, gray and the palette sets;
# ctest and the GPU unit tests (none of them runs k_recon).  ~/results/gate.txt, ~/results/ctest.txt
B=~/new/build/av1gpu; G=~/new/tests/decoder_gate.py
mkdir -p ~/sets/gate30 ~/sets/gate_thumbs
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/sets/gate30/; done
ls ~/sets/thumbs/*.jpg | awk 'NR % 10 == 3' | while read f; do ln -sf $(readlink -f $f) ~/sets/gate_thumbs/; done
R=~/results/gate.txt; : > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 8 "$@" > ~/results/gate_$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate_$n.log >> $R; }
{
gate photos30_q60_120_200 --images ~/sets/gate30 --q 60,120,200
gate photos30_444_10_q120 --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
gate photos30_q13_26 --images ~/sets/gate30 --q 13,26
gate photos30_slow --images ~/sets/gate30 -- --preset slow
gate thumbs_q60_120 --images ~/sets/gate_thumbs --q 60,120
gate odd_q120 --images ~/sets/odd
gate odd_444_10 --images ~/sets/odd --expect-depth 10 -- --yuv 444 --depth 10
gate gray_q120 --images ~/sets/gray
gate mono_q26_120 --images ~/sets/mono --q 26,120
gate rgba --images ~/data/rgba/set --expect-alpha
for ps in screen flat_png flat_jpg; do
  gate pal_$ps --images ~/pal/sets/$ps
  gate pal_${ps}_on --images ~/pal/sets/$ps -- --palette on
done
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
