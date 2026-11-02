#!/bin/bash
# pipe.sh: after setup.sh.  SASS of both trees (CPU only) next to the identity matrix; then the speed A/B alone on
# the GPU; then ctest and memcheck / synccheck (both binaries); then racecheck (both binaries, default and
# --yuv 444 --depth 10, in parallel).  ~/results/pipe.log; PIPE_DONE at the end.
L=~/results/pipe.log; exec >> $L 2>&1
echo "start $(date -u +%T)"
( python3 ~/sass.py old 1374 > ~/results/sass_old.log 2>&1; python3 ~/sass.py new 1375 > ~/results/sass_new.log 2>&1
  echo "sass done $(date -u +%T)" ) &
SP=$!
bash ~/identity.sh
echo "identity done $(date -u +%T): $(grep -cE 'differ 0, .*rc 0/0' ~/results/identity.txt) of $(grep -cE '^[a-z0-9_]+: inputs' ~/results/identity.txt) sets with 0 differences"
wait $SP
echo "GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
bash ~/speed.sh race noise
echo "speed done $(date -u +%T)"
( cd ~/new/build && t=$(date +%s) && ctest --output-on-failure -j 4 > ~/results/ctest.txt 2>&1
  echo "ctest rc $? ($(( $(date +%s) - t )) s)" >> ~/results/ctest.txt ) &
( bash ~/sanitizer.sh ~/new/build/av1gpu _new; bash ~/sanitizer.sh ~/old/build/av1gpu _old ) &
wait
echo "ctest + memcheck/synccheck done $(date -u +%T)"
bash ~/race_ab.sh new old
python3 ~/race_sum.py ~/results/race/new_0.log ~/results/race/new_1.log ~/results/race/old_0.log ~/results/race/old_1.log \
  > ~/results/race_summary.txt 2>&1
echo "racecheck done $(date -u +%T)"
echo PIPE_DONE
