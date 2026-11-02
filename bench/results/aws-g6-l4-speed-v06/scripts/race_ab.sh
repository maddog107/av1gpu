#!/bin/bash
# race_ab.sh TAG...: compute-sanitizer racecheck in its default analysis mode (one report per pair of source
# locations, with the hazard count) on ~/sets/san3, default and --yuv 444 --depth 10, for each build ~/TAG
# (new, old, or a bisect tree); in parallel.  ~/results/race/TAG_K.log; ~/results/race.txt = the distinct pairs.
CS=/usr/local/cuda/bin/compute-sanitizer
mkdir -p ~/results/race
for T in "$@"; do
  for k in ${KS:-0 1}; do
    o=""; [ $k = 1 ] && o="--yuv 444 --depth 10"
    ( rm -rf ~/work/race_${T}_$k; s=$(date +%s)
      timeout 2700 $CS --tool racecheck --print-limit 10000 ~/$T/build/av1gpu --batch ~/sets/san3 --outdir ~/work/race_${T}_$k $o \
        > ~/results/race/${T}_$k.log 2>&1
      echo "rc $? $(( $(date +%s) - s )) s" >> ~/results/race/${T}_$k.log ) &
  done
done
wait
{ for T in "$@"; do for k in ${KS:-0 1}; do
    f=~/results/race/${T}_$k.log
    echo "== $T ($(cat ~/$T/COMMIT | cut -c1-7)) $([ $k = 1 ] && echo '--yuv 444 --depth 10' || echo default): $(grep -E 'RACECHECK SUMMARY' $f) $(tail -n 1 $f)"
    grep --no-group-separator -A 1 -E 'Race reported' $f |
      sed -E 's/\+0x[0-9a-f]+//g; s/ at 0x[0-9a-f]+//g; s/=========//g; s/ +/ /g' | paste -d ' ' - -
  done; done; } > ~/results/race_$(echo "$@" | tr ' ' '_').txt 2>&1
echo RACE_DONE >> ~/results/race_$(echo "$@" | tr ' ' '_').txt
