#!/bin/bash
# checks9.sh BIN: with BIN (main c553ffb's CUDA 13 package, B13) only.
#  1. decoder gate (tests/decoder_gate.py of c553ffb, dav1d + aomdec): 30 size-range photos at qindex 60/120/200 and
#     4:4:4 10-bit; 28 thumbnails; RGBA; odd sizes; monochrome; Kodak lossless; the palette sets (default, --palette on,
#     4:4:4 10-bit, screen lossless).  ~/results/gate.txt, logs ~/results/gate/
#  2. compute-sanitizer memcheck (--leak-check full), racecheck (analysis mode) and synccheck: --batch on speed-v06's
#     3 photos (default and --yuv 444 --depth 10) and on the production pair; single-file mode on a 2.5 MP photo, its
#     512 x 384 thumbnail, a 320 x 320 thumbnail, a screen PNG (palette) and a Kodak PNG with --lossless.  Every output
#     must equal the plain run's.  ~/results/sanitizer.txt, logs ~/results/san/, racecheck pairs ~/results/race_pairs.txt
# Prints CHECKS_DONE.
B=$1; MODE=${2:-all}; G=~/src/tests/decoder_gate.py; CS=/usr/local/cuda/bin/compute-sanitizer
mkdir -p ~/gs/photos30 ~/gs/thumbs28 ~/results/gate
ls ~/images/*.jpg | awk 'NR % 10 == 1 || NR % 10 == 4 || NR % 10 == 7' | while read f; do ln -sf $f ~/gs/photos30/; done
ls ~/sets/thumbs/*.jpg | awk 'NR % 10 == 3' | while read f; do ln -sf $(readlink -f $f) ~/gs/thumbs28/; done
if [ $MODE != san ]; then
R=~/results/gate.txt; echo "BIN $B ($(sha256sum < $B | cut -c1-16)); dav1d $(dav1d --version); $(aomdec --help 2>&1 | grep -m1 -io 'v[0-9.]*')" > $R
gate() { local n=$1; shift; local t=$(date +%s); echo "== $n: $*" >> $R
  python3 $G --bin $B --jobs 16 --json ~/results/gate/$n.json "$@" > ~/results/gate/$n.log 2>&1; echo "rc $? ($(( $(date +%s) - t )) s)" >> $R
  tail -n 3 ~/results/gate/$n.log >> $R; }
gate photos30_q60_120_200 --images ~/gs/photos30 --q 60,120,200
gate photos30_444_10_q120 --images ~/gs/photos30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
gate thumbs28_q60_120 --images ~/gs/thumbs28 --q 60,120
gate rgba --images ~/data/rgba/set --expect-alpha
gate odd_q120 --images ~/sets/odd
gate mono_q26_120 --images ~/sets/mono --q 26,120
gate kodak_lossless --images ~/ll/kodak --lossless
for s in screen flat_png flat_jpg; do
  gate pal_${s} --images ~/pal/sets/$s
  gate pal_${s}_on --images ~/pal/sets/$s -- --palette on
  gate pal_${s}_444_10 --images ~/pal/sets/$s --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
done
gate pal_screen_lossless --images ~/pal/sets/screen --lossless
gate pal_flat_png_lossless --images ~/pal/sets/flat_png --lossless
echo GATE_DONE >> $R
fi
[ $MODE = gate ] && { echo CHECKS_DONE; exit 0; }
# ---- sanitizers
D=~/gs/san3; rm -rf $D; mkdir -p $D
python3 - <<'EOF' > ~/results/san3.txt
import glob, os
from PIL import Image
fs = sorted(glob.glob(os.path.expanduser("~/images/*.jpg")))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
pick = [min((f for f in fs if mp[f] >= 2), key=mp.get), min((f for f in fs if 1 <= mp[f] < 2), key=lambda f: abs(mp[f] - 1.5)),
        min((f for f in fs if 0.5 <= mp[f] < 1), key=lambda f: abs(mp[f] - 0.75))]
for f in pick:
    os.symlink(f, os.path.expanduser("~/gs/san3/") + os.path.basename(f))
    print(os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
EOF
SCR=$(ls ~/pal/sets/screen/* | head -1)
L=~/results/san; W=~/work/san; rm -rf $L $W; mkdir -p $L $W
S=~/results/sanitizer.txt; { echo "BIN $B ($(sha256sum < $B | cut -c1-16)); $($CS --version | tail -1)"; cat ~/results/san3.txt; echo "screen PNG: $(basename $SCR)"; } > $S
# job specs: name|mode|input|options
J=$W/jobs.txt; : > $J
echo "san3|batch|$D|" >> $J; echo "san3_444_10|batch|$D|--yuv 444 --depth 10" >> $J; echo "pair|batch|$HOME/st/pair.txt|" >> $J
echo "photo|single|$HOME/st/photo.jpg|" >> $J; echo "thumb512|single|$HOME/st/thumb512.jpg|" >> $J; echo "t320|single|$HOME/st/t320.jpg|" >> $J
echo "screen|single|$SCR|" >> $J; echo "kodim01_lossless|single|$HOME/ll/kodak/kodim01.png|--lossless" >> $J
outp() { [ $2 = batch ] && echo "$W/$1_$3" || echo "$W/$1_$3.avif"; }
while IFS='|' read -r n m in o; do   # plain reference runs
  if [ $m = batch ]; then $B --batch $in --outdir $(outp $n $m ref) $o > $L/${n}_ref.log 2>&1; else $B $in $(outp $n $m ref) $o > $L/${n}_ref.log 2>&1; fi
done < $J
one() {   # tool name mode input options
  local tool=$1 n=$2 m=$3 in=$4; shift 4; local out=$(outp $n $m $tool) t=$(date +%s) extra=""
  [ $tool = memcheck ] && extra="--leak-check full"; [ $tool = racecheck ] && extra="--print-limit 10000"
  if [ $m = batch ]; then timeout 3600 $CS --tool $tool $extra --error-exitcode 99 $B --batch $in --outdir $out "$@" > $L/${n}_$tool.log 2>&1
  else timeout 3600 $CS --tool $tool $extra --error-exitcode 99 $B $in $out "$@" > $L/${n}_$tool.log 2>&1; fi
  local rc=$? d=0 nf=0
  if [ $m = batch ]; then for f in $(outp $n $m ref)/*; do nf=$((nf + 1)); cmp -s $f $out/$(basename $f) || d=$((d + 1)); done
  else nf=1; cmp -s $(outp $n $m ref) $out || d=1; fi
  echo "$tool $n [${*:-default}]: rc $rc, $(( $(date +%s) - t )) s, outputs differing from the plain run: $d of $nf; $(grep -aE 'SUMMARY' $L/${n}_$tool.log | tail -1 | sed 's/=========//')"
}
export -f one outp; export B CS L W
while IFS='|' read -r n m in o; do for tool in memcheck synccheck racecheck; do echo "$tool|$n|$m|$in|$o"; done; done < $J |
  xargs -P 4 -I{} bash -c 'IFS="|" read -r tool n m in o <<< "{}"; one $tool $n $m $in $o' >> $S
python3 ~/src/bench/results/aws-g6-l4-speed-v06/scripts/race_sum.py $L/*_racecheck.log > ~/results/race_pairs.txt 2>&1
echo "racecheck pairs outside k_entropy_win / vload / vstore: $(grep -E '^ +[0-9]+ ' ~/results/race_pairs.txt | grep -vcE 'k_entropy_win|vload|vstore')" >> $S
echo SANITIZER_DONE >> $S
echo CHECKS_DONE
