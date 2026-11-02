#!/bin/bash
# speed.sh PHASE...: nvidia-smi dmon (power, utilisation, clocks; 1 s) around spd.py PHASE...; ~/results/main.log
cd ~
nvidia-smi dmon -s puc -o DT >> ~/results/dmon.txt 2>&1 &
DM=$!
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv >> ~/results/clocks_start.txt 2>&1
python3 ~/spd.py "$@" >> ~/results/main.log 2>&1
echo "spd rc $?" >> ~/results/main.log
kill $DM
echo MAIN_DONE >> ~/results/main.log
