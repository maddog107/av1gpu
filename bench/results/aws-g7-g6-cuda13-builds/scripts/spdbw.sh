#!/bin/bash
# spdbw.sh BIN SET ROUNDS: encode speed of BIN on SET (a directory) with
#   A  the instance's driver (595) as installed
#   B  the 13.4 UMD (615.71.10), recompile on (default), warm compute cache (re-finalized kernels)
#   C  the 13.4 UMD, CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=DISABLED, cache disabled (driver-patched kernels)
# each: a warm-up, then 3 timed runs; ROUNDS rounds, config order rotating.  Outputs of every config's last run
# kept in ~/spd/out_<cfg>, then compared with cmp.
BIN=$1; SET=$2; R=${3:-2}
U=/usr/local/cuda-13.4/compat
mkdir -p ~/spd; cd ~/spd
rm -rf cacheB; LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$PWD/cacheB $BIN --batch ~/a.jpg --outdir /tmp/o1 > /dev/null 2>&1   # warm B's cache
echo "cacheB $(du -sm cacheB | cut -f1) MB"
run() {  # cfg
  case $1 in
    A) env -u LD_LIBRARY_PATH CUDA_CACHE_DISABLE=1 "${@:2}" ;;
    B) env LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$PWD/cacheB "${@:2}" ;;
    C) env LD_LIBRARY_PATH=$U CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=DISABLED CUDA_CACHE_DISABLE=1 "${@:2}" ;;
  esac
}
for r in $(seq $R); do
  case $r in 1) O="A B C";; 2) O="C B A";; *) O="B A C";; esac
  for c in $O; do
    rm -rf out_$c; run $c $BIN --batch $SET --outdir out_$c > /dev/null 2>&1    # warm-up
    for i in 1 2 3; do
      rm -rf out_$c; s=$(date +%s.%N)
      l=$(run $c $BIN --batch $SET --outdir out_$c 2>&1 | grep "MP/s" | head -1)
      e=$(date +%s.%N)
      echo "round $r cfg $c run $i wall $(printf %.2f $(echo "$e - $s" | bc)) s | $(echo "$l" | grep -oE '[0-9.]+ MP/s' | head -1) | files $(ls out_$c | wc -l)"
    done
  done
done
for c in B C; do
  n=0; d=0; for f in out_A/*; do n=$((n+1)); cmp -s $f out_$c/$(basename $f) || d=$((d+1)); done
  echo "cmp A vs $c: $n files, $d differ"
done
