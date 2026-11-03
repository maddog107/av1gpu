#!/bin/bash
# run_speed.sh: waits for ident_single.sh to finish (the GPU to itself), then spd8.py single + batch with nvidia-smi
# dmon (1 s) around them; ~/results/speed.log, ~/results/dmon.txt.  Prints SPEED_DONE.
until grep -q IDENT_SINGLE_DONE ~/results/ident_single.txt 2>/dev/null; do sleep 5; done
cd ~
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_start.txt 2>&1
nvidia-smi dmon -s puc -o DT > ~/results/dmon.txt 2>&1 &
DM=$!
echo "start $(date -u +%H:%M:%S)" >> ~/results/speed.log
python3 ~/spd8.py single batch >> ~/results/speed.log 2>&1
echo "spd8 rc $? end $(date -u +%H:%M:%S)" >> ~/results/speed.log
kill $DM
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv > ~/results/clocks_end.txt 2>&1
echo SPEED_DONE >> ~/results/speed.log
