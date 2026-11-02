#!/bin/bash
# startup.sh BIN LABEL [REPS]: wall time (s) of a one-image --batch run (~/a.jpg, 1600x1200) with the CUDA compute
# cache empty, warm (the same cache again) and disabled; cache size after the empty-cache run.
BIN=$1; L=$2; R=${3:-2}
W=~/st/$L; rm -rf $W; mkdir -p $W; cd $W
printf '%s\n' ~/a.jpg > list.txt
cat $BIN > /dev/null                      # binary in the page cache: disk reads are not part of the times
t() { local s=$(date +%s.%N); "$@" > log.txt 2>&1; local rc=$?; local e=$(date +%s.%N); printf '%.2f s rc=%d' $(echo "$e - $s" | bc) $rc; }
for r in $(seq $R); do
  rm -rf cache out; echo "$L empty    $(t env CUDA_CACHE_PATH=$W/cache $BIN --batch list.txt --outdir out) cache=$(du -sm cache 2>/dev/null | cut -f1) MB"
  rm -rf out;       echo "$L warm     $(t env CUDA_CACHE_PATH=$W/cache $BIN --batch list.txt --outdir out)"
  rm -rf out;       echo "$L disabled $(t env CUDA_CACHE_DISABLE=1 $BIN --batch list.txt --outdir out)"
done
