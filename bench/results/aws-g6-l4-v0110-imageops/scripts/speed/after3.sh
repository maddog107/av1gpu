#!/bin/bash
# after the first chain: the bordered set (cuts really happen), its traces, and the kernels alone on one stream
B=$HOME/pr/build/av1gpu; L=~/results/after.log
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
until grep -q AFTER_DONE $L; do sleep 5; done
python3 ~/mk_border.py > ~/results/border.txt 2>&1
echo "$(date -u +%H:%M:%S) on2" >> $L
python3 ~/spd65.py on2 8 "plain=$B;plain2=$B;never=$B|--trim --fuzz 10% --trim-limits 100%,100%;trim0=$B|--trim --fuzz 0%;trim10=$B|--trim --fuzz 10%;conv=$B|$CONV" -- border b444 > ~/results/spd_on2.txt 2>&1
echo "$(date -u +%H:%M:%S) prof border" >> $L; ONLY_BORDER=1 bash ~/prof65.sh $B prb > ~/results/prof_prb.txt 2>&1
echo "$(date -u +%H:%M:%S) prof1" >> $L; bash ~/prof1.sh $B pr > ~/results/prof1_pr.txt 2>&1
echo "$(date -u +%H:%M:%S) AFTER3_DONE" >> $L
