#!/bin/bash
# after the first chain: the longer option-off rerun on the three sets with the widest intervals, then module and JIT figures
A=$HOME/main/build/av1gpu; C=$HOME/bin/main_copy; B=$HOME/pr/build/av1gpu; L=~/results/after.log
until grep -q AFTER_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) off2" >> $L; python3 ~/spd65.py off2 24 "A=$A;C=$C;B=$B" -- std thumbs pngwebp > ~/results/spd_off2.txt 2>&1
echo "$(date -u +%H:%M:%S) rel build" >> $L; bash ~/rel65.sh > ~/results/rel65.out 2>&1
echo "$(date -u +%H:%M:%S) mods" >> $L; bash ~/mods65.sh > ~/results/mods65.out 2>&1
echo "$(date -u +%H:%M:%S) AFTER2_DONE" >> $L
