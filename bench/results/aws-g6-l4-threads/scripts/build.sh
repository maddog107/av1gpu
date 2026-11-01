#!/bin/bash
# On the instance: sm_89 build of main, then incremental rebuilds of perf/cpu-threads and contrib/batch-fallback
# -> ~/bin/av1gpu-{main,new,bf}
R=~/results
cd ~/av1gpu
git checkout -q main
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > $R/cfg.log 2>&1
for b in main perf/cpu-threads contrib/batch-fallback; do
  git checkout -q $b
  t0=$(date +%s); ninja -C build > $R/build_$(basename $b).log 2>&1; rc=$?
  echo "$b: exit $rc, $(( $(date +%s) - t0 )) s"
  case $b in main) n=main;; perf/cpu-threads) n=new;; contrib/batch-fallback) n=bf;; esac
  [ $rc = 0 ] && cp build/av1gpu ~/bin/av1gpu-$n
done
git checkout -q main
md5sum ~/bin/av1gpu-*
echo BUILD_ALL_DONE
