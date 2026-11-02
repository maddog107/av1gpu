#!/bin/bash
# spdab.sh BIN_A BIN_B SET [ROUNDS] [ENV...]: encode speed of two binaries on SET (directory), a warm-up then 3 timed
# runs each, ROUNDS rounds (default 2) with the order alternating; ENV (VAR=value ...) applies to both.  Prints every
# run's MP/s (the batch summary line) and wall time, then the medians.
A=$1; B=$2; SET=$3; R=${4:-2}; shift 4
mkdir -p ~/spd; cd ~/spd
declare -A V
for r in $(seq $R); do
  [ $((r % 2)) = 1 ] && O="a b" || O="b a"
  for x in $O; do
    [ $x = a ] && bin=$A || bin=$B
    rm -rf o_$x; env "$@" $bin --batch $SET --outdir o_$x > /dev/null 2>&1
    for i in 1 2 3; do
      rm -rf o_$x; s=$(date +%s.%N)
      m=$(env "$@" $bin --batch $SET --outdir o_$x 2>&1 | grep "MP/s" | head -1 | grep -oE '[0-9.]+ MP/s' | head -1 | cut -d' ' -f1)
      w=$(printf %.2f $(echo "$(date +%s.%N) - $s" | bc))
      echo "round $r $x run $i: $m MP/s, wall $w s, $(ls o_$x | wc -l) files"
      V[$x]="${V[$x]} $m"
    done
  done
done
for x in a b; do echo "median $x: $(echo ${V[$x]} | tr ' ' '\n' | sort -n | awk '{v[NR]=$1} END {print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}') MP/s (${V[$x]} )"; done
