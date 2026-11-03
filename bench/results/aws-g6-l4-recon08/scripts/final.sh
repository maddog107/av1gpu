#!/bin/bash
# final.sh: the verification of ~/new (perf/recon) against ~/old (main cde1f74): full build with tests; byte identity
# (identity8.sh) with racecheck + memcheck alongside; then synccheck, the decoder gate, ctest and the GPU tests; then
# the speed A/B alone on the GPU (spd8.py: main, perf/recon, and main again as the noise run).  ~/results/final.log
L=~/results/final.log; exec >> $L 2>&1
echo "start $(date -u +%T)"
echo "${TAG:-abf26f5}" > ~/new/COMMIT; echo "cde1f74" > ~/old/COMMIT
cmake --build ~/new/build -j 24 > ~/r/final_build.log 2>&1; echo "build rc $? $(date -u +%T)"
mkdir -p ~/bins; cp ~/old/build/av1gpu ~/bins/main_copy; cp ~/new/build/av1gpu ~/bins/final
cp ~/r/identity8.sh ~/identity8.sh
( bash ~/r/resusage.sh; echo "resusage done $(date -u +%T)" ) &
UP=$!
( bash ~/identity8.sh; echo "identity rc $? $(date -u +%T)" ) &
IP=$!
( cd ~ && KS="0 1" bash ~/race_ab.sh new; echo "race rc $? $(date -u +%T)" ) &
RP=$!
( B=~/new/build/av1gpu; CS=/usr/local/cuda/bin/compute-sanitizer; mkdir -p ~/results/memcheck ~/work/mc
  python3 - <<'EOF'
import glob, os
from PIL import Image
d = os.path.expanduser("~/sets/san3"); os.makedirs(d, exist_ok=True)
fs = sorted(glob.glob(os.path.expanduser("~/images/*.jpg")))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
pick = [min((f for f in fs if mp[f] >= 2), key=mp.get), min((f for f in fs if 1 <= mp[f] < 2), key=lambda f: abs(mp[f] - 1.5)),
        min((f for f in fs if 0.5 <= mp[f] < 1), key=lambda f: abs(mp[f] - 0.75))]
for f in pick:
    l = os.path.join(d, os.path.basename(f))
    if not os.path.exists(l): os.symlink(f, l)
    print(os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
EOF
  for k in 0 1; do o=""; [ $k = 1 ] && o="--yuv 444 --depth 10"
    ( s=$(date +%s); timeout 2700 $CS --tool memcheck --print-limit 100 --error-exitcode 99 $B --batch ~/sets/san3 --outdir ~/work/mc/m$k $o \
        > ~/results/memcheck/memcheck_$k.log 2>&1; echo "memcheck [$o] rc $? $(( $(date +%s) - s )) s: $(grep SUMMARY ~/results/memcheck/memcheck_$k.log)" ) &
  done; wait ) > ~/results/memcheck.txt 2>&1 &
MP=$!
wait $IP; echo "identity done $(date -u +%T)"
( B=~/new/build/av1gpu; CS=/usr/local/cuda/bin/compute-sanitizer; mkdir -p ~/results/synccheck ~/work/sc
  for k in 0 1; do o=""; [ $k = 1 ] && o="--yuv 444 --depth 10"
    ( s=$(date +%s); timeout 2700 $CS --tool synccheck --print-limit 100 --error-exitcode 99 $B --batch ~/sets/san3 --outdir ~/work/sc/s$k $o \
        > ~/results/synccheck/synccheck_$k.log 2>&1; echo "synccheck [$o] rc $? $(( $(date +%s) - s )) s: $(grep SUMMARY ~/results/synccheck/synccheck_$k.log)" ) &
  done; wait ) > ~/results/synccheck.txt 2>&1 &
SP=$!
bash ~/r/checks8.sh; echo "checks done $(date -u +%T)"
wait $MP $SP $RP $UP; echo "sanitizers done $(date -u +%T)"
grep -c "differ 0" ~/results/identity.txt; grep -E "DIFF|ONLY_B" ~/results/identity.txt | head; grep -vE "differ 0, .*rc 0/0|differ 0, failed runs 0|^A=|took|IDENTITY_DONE" ~/results/identity.txt
echo "speed start $(date -u +%T)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv
nvidia-smi --query-compute-apps=pid,name --format=csv
TAG=final SETS=sr,std,thumbs,single,single2 python3 ~/r/spd8.py main=$HOME/bins/main_copy recon08=$HOME/bins/final main_noise=$HOME/old/build/av1gpu > ~/results/speed_final.txt 2>&1
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv
echo "FINAL_DONE $(date -u +%T)"
