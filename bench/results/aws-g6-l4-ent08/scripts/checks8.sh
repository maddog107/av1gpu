#!/bin/bash
# checks8.sh: the branch (~/br) only.  Decoder gate (dav1d + aomdec, N/N) on photos, thumbnails, odd sizes, gray, RGBA,
# Kodak lossless and the palette sets; ctest and gpu_fuzz_test; compute-sanitizer memcheck + synccheck on 2 photos
# (a 4 x 4-tile turbo photo >= 2 MP and a base-preset photo < 1 MP) and racecheck (analysis mode) on 2 thumbnails,
# main and branch.  ~/results/gate.txt, ctest.txt, sanitizer.txt (logs next to them).
B=~/br/build/av1gpu; G=~/br/tests/decoder_gate.py; CS=/usr/local/cuda/bin/compute-sanitizer
mkdir -p ~/sets/gate30 ~/sets/gate_thumbs
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/sets/gate30/; done
ls ~/sets/thumbs/*.jpg | awk 'NR % 10 == 3' | while read f; do ln -sf $(readlink -f $f) ~/sets/gate_thumbs/; done
R=~/results/gate.txt; : > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 8 "$@" > ~/results/gate_$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate_$n.log >> $R; }
gate photos30_q60_120_200 --images ~/sets/gate30 --q 60,120,200
gate photos30_444_10_q120 --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
gate photos30_q13 --images ~/sets/gate30 --q 13
gate thumbs_q60_120 --images ~/sets/gate_thumbs --q 60,120
gate odd_q120 --images ~/sets/odd
gate gray_q120 --images ~/sets/gray
gate rgba --images ~/data/rgba/set --expect-alpha
gate kodak_lossless --images ~/ll/kodak --lossless
for s in screen flat_png flat_jpg; do
  gate pal_${s} --images ~/pal/sets/$s
  gate pal_${s}_lossless --images ~/pal/sets/$s --lossless
done
echo GATE_DONE >> $R
C=~/results/ctest.txt; : > $C
{ cd ~/br/build && t=$(date +%s) && ctest --output-on-failure -j 4; echo "ctest rc $? ($(( $(date +%s) - t )) s)"
  ./gpu_fuzz_test --seed 1 --count 400; echo "gpu_fuzz_test rc $?"
  ./gpu_fuzz_test --seed 77 --count 200 --yuv444 60 --depth10 60 --lossless 30; echo "gpu_fuzz_test (444/10/ll) rc $?"; } >> $C 2>&1
echo CTEST_DONE >> $C
# sanitizers
D=~/sets/san2; rm -rf $D; mkdir -p $D
python3 - <<'EOF'
import glob, os
from PIL import Image
fs = sorted(glob.glob(os.path.expanduser("~/images/*.jpg")))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
for f in (min((f for f in fs if mp[f] >= 2), key=mp.get), min((f for f in fs if 0.5 <= mp[f] < 1), key=lambda f: abs(mp[f] - 0.75))):
    os.symlink(f, os.path.expanduser("~/sets/san2/") + os.path.basename(f))
EOF
D2=~/sets/san_thumbs; rm -rf $D2; mkdir -p $D2
ls ~/sets/thumbs/d512_*.jpg | head -2 | while read f; do ln -sf $(readlink -f $f) $D2/; done
S=~/results/sanitizer.txt; L=~/results/san; mkdir -p $L ~/work/san; : > $S
ls -l $D $D2 | awk '{print $NF}' >> $S
rm -rf ~/work/san/ref; $B --batch $D --outdir ~/work/san/ref > $L/ref.log 2>&1
san() { local tag=$1 bin=$2 dir=$3; shift 3; local out=~/work/san/$tag; rm -rf $out; local t=$(date +%s)
  timeout 1500 $CS "$@" --print-limit 100 $bin --batch $dir --outdir $out > $L/$tag.log 2>&1
  echo "$tag rc $?, $(( $(date +%s) - t )) s: $(grep -E 'SUMMARY' $L/$tag.log | tail -1)" >> $S; }
san memcheck_br $B $D --tool memcheck --error-exitcode 99 &
san synccheck_br $B $D --tool synccheck --error-exitcode 99 &
san racecheck_br $B $D2 --tool racecheck --racecheck-report analysis &
san racecheck_main ~/main/build/av1gpu $D2 --tool racecheck --racecheck-report analysis &
wait
d=0; for f in ~/work/san/ref/*; do cmp -s $f ~/work/san/memcheck_br/$(basename $f) || d=$((d + 1)); done
echo "memcheck outputs differing from the plain run: $d" >> $S
for t in racecheck_br racecheck_main; do
  echo "== $t hazards per source pair" >> $S
  grep -E 'Error: (Read|Write)|Warning: (Read|Write)' -A2 $L/$t.log | grep -oE 'in (\S+):[0-9]+' | paste - - | sort | uniq -c | sort -rn | head -20 >> $S
done
echo SANITIZER_DONE >> $S
