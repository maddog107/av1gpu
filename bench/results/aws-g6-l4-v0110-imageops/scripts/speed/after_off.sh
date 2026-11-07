#!/bin/bash
# runs after the option-off speed run; one stage at a time, nothing else on the GPU
A=$HOME/main/build/av1gpu; C=$HOME/bin/main_copy; B=$HOME/pr/build/av1gpu; L=~/results/after.log
until grep -q "SPD_DONE off\|Traceback" ~/results/spd_off.txt; do sleep 5; done
echo "$(date -u +%H:%M:%S) cold" >> $L;  python3 ~/cold65.py sm89 24 "A=$A,C=$C,B=$B" > ~/results/cold_sm89.txt 2>&1
echo "$(date -u +%H:%M:%S) mem" >> $L;   python3 ~/mem65.py sm89 3 "A=$A,C=$C,B=$B" > ~/results/mem_sm89.txt 2>&1
echo "$(date -u +%H:%M:%S) on" >> $L;    bash ~/on65.sh $B on 6 comics c444 sr std thumbs pngwebp > ~/results/spd_on.txt 2>&1
echo "$(date -u +%H:%M:%S) prof" >> $L;  bash ~/prof65.sh $B pr > ~/results/prof_pr.txt 2>&1
echo "$(date -u +%H:%M:%S) AFTER_DONE" >> $L
