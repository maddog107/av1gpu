#!/bin/bash
# sanitizer.sh [BIN] [TAG]: compute-sanitizer racecheck / memcheck / synccheck on 3 size-range photos (the smallest photo
# >= 2 MP: turbo with 4x4 tiles; one 1-2 MP: turbo, no tiles; one below 1 MP: the base small preset), default
# settings and --yuv 444 --depth 10, all six tool x option runs in parallel; each output must also equal the
# uninstrumented run's.  ~/results/sanitizer${TAG}.txt, full logs in ~/results/san${TAG}/.
B=${1:-~/new/build/av1gpu}; T=${2:-}
CS=/usr/local/cuda/bin/compute-sanitizer
D=~/sets/san3; rm -rf $D; mkdir -p $D
python3 - <<'EOF'
import glob, os
from PIL import Image
fs = sorted(glob.glob(os.path.expanduser("~/images/*.jpg")))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
pick = [min((f for f in fs if mp[f] >= 2), key=mp.get), min((f for f in fs if 1 <= mp[f] < 2), key=lambda f: abs(mp[f] - 1.5)),
        min((f for f in fs if 0.5 <= mp[f] < 1), key=lambda f: abs(mp[f] - 0.75))]
for f in pick:
    os.symlink(f, os.path.expanduser("~/sets/san3/") + os.path.basename(f))
    print(os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
EOF
L=~/results/san$T; rm -rf $L; mkdir -p $L ~/work/san$T
R=~/results/sanitizer$T.txt
{ echo "BIN $B ($(cat $(dirname $(dirname $B))/COMMIT))"; ls -l $D | tail -n +2 | awk '{print $NF}'; $CS --version | tail -1; } > $R
for k in 0 1; do
  o=""; [ $k = 1 ] && o="--yuv 444 --depth 10"
  rm -rf ~/work/san$T/ref$k; $B --batch $D --outdir ~/work/san$T/ref$k $o > $L/ref$k.log 2>&1
done
run() { local tool=$1 k=$2; shift 2; local o=""; [ $k = 1 ] && o="--yuv 444 --depth 10"
  local out=~/work/san$T/${tool}$k; rm -rf $out; local t=$(date +%s)
  timeout 2700 $CS --tool $tool "$@" --print-limit 100 --error-exitcode 99 $B --batch $D --outdir $out $o > $L/${tool}$k.log 2>&1
  local rc=$? d=0
  for f in ~/work/san$T/ref$k/*; do cmp -s $f $out/$(basename $f) || d=$((d + 1)); done
  echo "$tool [${o:-default}] rc $rc, $(( $(date +%s) - t )) s, outputs differing from the plain run: $d; $(grep -E 'SUMMARY' $L/${tool}$k.log | tail -1)"; }
for k in 0 1; do   # racecheck: race_ab.sh (analysis mode)
  run memcheck $k --leak-check full >> $R.$k.mem &
  run synccheck $k >> $R.$k.sync &
done
wait
cat $R.*.mem $R.*.sync >> $R; rm -f $R.*.mem $R.*.sync
echo SANITIZER_DONE >> $R
