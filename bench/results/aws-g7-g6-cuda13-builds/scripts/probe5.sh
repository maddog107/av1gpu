#!/bin/bash
# probe5.sh: CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE in the 13.4 UMD
U=/usr/local/cuda-13.4/compat/libcuda.so.615.71.10
cd ~/p2
echo "== strings near it"
strings -n 6 $U | grep -n -B8 -A8 "CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE" | head -60
strings -n 6 $U | grep -iE "recompil|refinaliz|re-finaliz|selective" | sort -u | head -40
echo "== small kernel (12.9, sm_120) with the variable"
for v in unset 0 1 2; do
  rm -rf c
  if [ $v = unset ]; then r=$(LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat CUDA_CACHE_PATH=$PWD/c ./v_129_120 2>&1)
  else r=$(LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=$v CUDA_CACHE_PATH=$PWD/c ./v_129_120 2>&1); fi
  echo "CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=$v: $r, cache entries $(find c -type f ! -name index 2>/dev/null | wc -l)"
done
echo "== v0.6.0, cache disabled, with the variable"
cd ~/st
for v in 0 1; do
  rm -rf out; s=$(date +%s.%N)
  LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=$v CUDA_CACHE_DISABLE=1 ~/rel/0.6.0/av1gpu --batch list.txt --outdir out > log.txt 2>&1
  echo "CUDA_SELECTIVE_DEVICE_CODE_RECOMPILE=$v: rc=$? $(echo "$(date +%s.%N) - $s" | bc) s"
done
