#!/bin/bash
# speed.sh PHASE...: nvidia-smi dmon (power, utilisation, clocks; 1 s) around spd.py PHASE...; ~/results/speed.log
cd ~
nvidia-smi dmon -s puc -o DT >> ~/results/dmon.txt 2>&1 &
DM=$!
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv >> ~/results/clocks_start.txt 2>&1
echo "GPU apps before: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> ~/results/speed.log
python3 ~/spd.py "$@" >> ~/results/speed.log 2>&1
echo "spd rc $?" >> ~/results/speed.log
kill $DM
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv >> ~/results/clocks_end.txt 2>&1
echo SPEED_DONE >> ~/results/speed.log
