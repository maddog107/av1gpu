#!/bin/bash
# verify2.sh: the final check of the branch head (~/new2: changes 1 + 2) against main (~/old): ~/new2 rebuilt with the
# tests; the full identity list (identity.sh, identity_pal.sh) with the checks (checks2.sh new2) alongside; then, with
# the GPU to itself, the 3-way speed A/B (new2 = 1 + 2, new = change 1, old = main), the noise run (main against
# main) and the single-JPEG cold-process times -> ~/results/{identity_new2,identity_pal_new2}.txt, ~/results/speed2.log
cd ~
cmake -S ~/new2 -B ~/new2/build -DAV1GPU_TESTS=ON > ~/results/new2_tests.log 2>&1
cmake --build ~/new2/build -j 24 >> ~/results/new2_tests.log 2>&1; echo "tests build rc $?" >> ~/results/new2_tests.log
( bash ~/checks2.sh new2 > ~/results/checks2.out 2>&1 ) &
CK=$!
bash ~/identity.sh ~/old/build/av1gpu ~/new2/build/av1gpu new2
bash ~/identity_pal.sh ~/old/build/av1gpu ~/new2/build/av1gpu new2
wait $CK
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_start2.txt 2>&1
( nvidia-smi dmon -s puc -o DT > ~/results/dmon2.txt 2>&1 ) &
DM=$!
SB="new2 new old" python3 ~/spd_an.py ab3 noise single > ~/results/speed2.log 2>&1
echo "spd rc $?" >> ~/results/speed2.log
kill $DM
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_end2.txt 2>&1
echo VERIFY2_DONE >> ~/results/speed2.log
