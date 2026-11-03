#!/bin/bash
# run_split.sh: after run_speed.sh (SPEED_DONE), spd8.py split (the size range with the trees after commits 1-3)
until grep -q SPEED_DONE ~/results/speed.log 2>/dev/null; do sleep 3; done
cd ~ && python3 ~/spd8_split.py split >> ~/results/speed.log 2>&1
echo SPLIT_DONE >> ~/results/speed.log
