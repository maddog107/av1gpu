#!/bin/bash
# verify.sh: the full byte-identity list (identity.sh: speed-v06's sets; identity_pal.sh: the palette sets) of ~/new
# against ~/old, with the checks (checks_an.sh) alongside; then, with the GPU to itself, the speed A/B, the noise run
# and the single-JPEG cold-process times (spd_an.py) -> ~/results/{identity,identity_pal}.txt, ~/results/speed.log
cd ~
( bash ~/checks_an.sh > ~/results/checks.out 2>&1 ) &
CK=$!
bash ~/identity.sh ~/old/build/av1gpu ~/new/build/av1gpu
bash ~/identity_pal.sh ~/old/build/av1gpu ~/new/build/av1gpu
wait $CK
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_start.txt 2>&1
python3 ~/spd_an.py ab noise single > ~/results/speed.log 2>&1
echo "spd rc $?" >> ~/results/speed.log
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_end.txt 2>&1
echo VERIFY_DONE >> ~/results/speed.log
