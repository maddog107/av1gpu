#!/bin/bash
# checks2.sh: this branch (~/new) only.  Decoder gate (dav1d + aomdec) on photos, thumbnails, odd sizes, RGBA and the
# palette sets; ctest and the GPU unit tests; compute-sanitizer memcheck on single-file mode and a short batch.
# ~/results/gate.txt, ~/results/ctest.txt, ~/results/memcheck.txt.  Prints CHECKS_DONE in each.
B=~/new/build/av1gpu; G=~/new/tests/decoder_gate.py
mkdir -p ~/sets/gate30 ~/sets/gate_thumbs
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/sets/gate30/; done
ls ~/sets/thumbs/*.jpg | awk 'NR % 10 == 3' | while read f; do ln -sf $(readlink -f $f) ~/sets/gate_thumbs/; done
R=~/results/gate.txt; : > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 16 "$@" > ~/results/gate_$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate_$n.log >> $R; }
(
gate photos30_q60_120_200 --images ~/sets/gate30 --q 60,120,200
gate photos30_444_10_q120 --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
gate thumbs_q60_120 --images ~/sets/gate_thumbs --q 60,120
gate rgba --images ~/data/rgba/set --expect-alpha
gate odd_q120 --images ~/sets/odd
gate pal_screen --images ~/pal/sets/screen
gate pal_screen_on --images ~/pal/sets/screen -- --palette on
gate pal_flat_png --images ~/pal/sets/flat_png
gate pal_flat_jpg --images ~/pal/sets/flat_jpg
gate pal_flat_jpg_on --images ~/pal/sets/flat_jpg -- --palette on
echo CHECKS_DONE >> $R ) &
C=~/results/ctest.txt; : > $C
( cd ~/new/build && t=$(date +%s) && ctest --output-on-failure -j 4; echo "ctest rc $? ($(( $(date +%s) - t )) s)"
  ./gpu_fuzz_test --seed 1 --count 400; echo "gpu_fuzz_test rc $?"
  ./gpu_postfilter_test --seed 1 --count 200; echo "gpu_postfilter_test rc $?"
  ./varboost_test; echo "varboost_test rc $?"; echo CHECKS_DONE ) >> $C 2>&1 &
wait
M=~/results/memcheck.txt; : > $M
CS=/usr/local/cuda/bin/compute-sanitizer
{ for f in ~/st/photo.jpg ~/st/t320.jpg ~/data/rgba/set/$(ls ~/data/rgba/set | head -1); do
    $CS --tool memcheck $B $f /tmp/mc.avif 2>&1 | tail -2 | sed "s#^#single $(basename $f): #"
  done
  $CS --tool memcheck $B $(ls ~/images/*_huge.jpg | head -1) /tmp/mc.avif --yuv 444 --depth 10 2>&1 | tail -2 | sed "s#^#single huge 444 10: #"
  mkdir -p /tmp/mcb; $CS --tool memcheck $B --batch ~/st/pair.txt --outdir /tmp/mcb 2>&1 | tail -2 | sed "s#^#batch pair: #"
  ls ~/images/*.jpg | head -3 > /tmp/mc3.txt; ls ~/images/*.jpg | head -12 > /tmp/mc12.txt
  $CS --tool memcheck $B --batch /tmp/mc3.txt --outdir /tmp/mcb 2>&1 | tail -2 | sed "s#^#batch 3 photos: #"
  $CS --tool memcheck $B --batch /tmp/mc12.txt --outdir /tmp/mcb 2>&1 | tail -2 | sed "s#^#batch 12 photos (32 queues): #"
  echo CHECKS_DONE; } > $M 2>&1
