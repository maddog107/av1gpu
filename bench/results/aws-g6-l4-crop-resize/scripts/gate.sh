#!/bin/bash
# gate.sh: tests/decoder_gate.py (dav1d + aomdec, N/N) on crop + resize runs of the branch, and ctest.  A crop of
# 65536x65536 at +1+1 / +3+1 is every image less its first row(s) / column(s): odd offsets on every input.
# ~/results/gate.txt, ~/results/ctest.txt.
B=~/cr/build/av1gpu; G=~/cr/tests/decoder_gate.py
mkdir -p ~/sets/gate30
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/sets/gate30/; done
R=~/results/gate.txt; : > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 16 "$@" > ~/results/gate_$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate_$n.log >> $R; }
{
gate photos30_420_8_crop_resize --images ~/sets/gate30 --q 60,120 --expect-format 420 --expect-depth 8 -- --crop 65536x65536+1+1 --max-size 1201x997
gate photos30_444_10_crop_resize --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10 --crop 65536x65536+3+1 --max-size 999x1201
gate photos30_slow_crop_resize --images ~/sets/gate30 --q 120 -- --preset slow --crop 65536x65536+1+1 --max-height 777
gate photos30_420_10_resize --images ~/sets/gate30 --q 120 --expect-depth 10 -- --depth 10 --max-size 1999x1999
gate rgba_crop_resize --images ~/data/rgba/set --q 60,120 --expect-alpha -- --crop 65536x65536+1+3 --max-size 257x257
gate rgba_444_10_resize --images ~/data/rgba/set --q 120 --expect-alpha --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10 --max-size 301x301
gate kodak_lossless_crop_resize --images ~/ll/kodak --lossless -- --crop 65536x65536+1+1 --max-size 501x333
gate kodak_lossless_ycgco_resize --images ~/ll/kodak --lossless -- --ycgco-re --max-size 400x400
gate rgba_lossless_resize --images ~/data/rgba/set --lossless --expect-alpha --alpha-lossless -- --max-size 300x300
gate odd_crop_resize --images ~/sets/odd --q 120 -- --crop 65536x65536+1+1 --max-size 40x40
gate gray_resize --images ~/sets/gray --q 120 -- --max-size 300x300
gate thumbs_resize --images ~/sets/thumbs --q 120 -- --max-size 100x100
} 2>&1
echo GATE_DONE >> $R
C=~/results/ctest.txt; : > $C
{ cd ~/cr/build && t=$(date +%s) && ctest --output-on-failure -j 4; echo "ctest rc $? ($(( $(date +%s) - t )) s)"; } >> $C 2>&1
echo CTEST_DONE >> $C
