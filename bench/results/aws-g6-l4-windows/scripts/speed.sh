#!/bin/bash
# speed.sh BIN TAG: warm-up + 3 runs per set (no outputs written) -> ~/res/TAG/speed.txt (set run mp_s cpu_cores)
BIN=$1; TAG=$2
cd ~/data
R=~/res/$TAG; mkdir -p $R; : > $R/speed.txt
for s in "jpg3.txt" "png40x2.txt" "mix3.txt" "mix3.txt --yuv 444 --depth 10"; do
  for k in 0 1 2 3; do
    line=$($BIN --batch $s --stats 2>/dev/null | grep '^{')
    mps=$(echo "$line" | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["mp_s"], d["cpu_cores"])')
    [ $k -gt 0 ] && echo "${s// /_} $k $mps" >> $R/speed.txt
  done
done
cat $R/speed.txt
