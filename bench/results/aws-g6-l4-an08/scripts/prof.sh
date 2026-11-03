#!/bin/bash
# prof.sh BIN TAG: stage-ms lines for the photo sets (one pass each, default settings), then ncu per-kernel GPU time
# (gpu__time_duration.sum, kernels serialised) for each set -> ~/results/prof/TAG_*.
BIN=${1:-~/old/build/av1gpu}; T=${2:-old}
R=~/results/prof; mkdir -p $R ~/work/prof
NCU="sudo /usr/local/cuda/bin/ncu"
for s in ${SETS:-sr:$HOME/images std:$HOME/images_std thumbs:$HOME/sets/thumbs}; do
  n=${s%%:*}; d=${s#*:}
  rm -rf ~/work/prof/$n; $BIN --batch $d --outdir ~/work/prof/$n --stats > $R/${T}_${n}_stage.txt 2>&1
done
for s in ${SETS:-sr:$HOME/images std:$HOME/images_std thumbs:$HOME/sets/thumbs}; do
  n=${s%%:*}; d=${s#*:}
  sudo rm -rf /tmp/ncu_${T}_$n
  $NCU --csv --metrics gpu__time_duration.sum --log-file $R/${T}_${n}_ncu.csv $BIN --batch $d --outdir /tmp/ncu_${T}_$n > $R/${T}_${n}_ncu.out 2>&1
  python3 ~/ksum.py $R/${T}_${n}_ncu.csv > $R/${T}_${n}_ksum.txt
done
echo PROF_DONE > $R/${T}_done
