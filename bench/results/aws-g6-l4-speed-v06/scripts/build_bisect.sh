#!/bin/bash
# build_bisect.sh TAG...: ~/TAG from ~/TAG.tar.gz, built like OLD (sm_89, tests off), two at a time; ~/results/setup/TAG.log
L=~/results/setup
b() { local v=$1; mkdir -p ~/$v; tar -xzf ~/$v.tar.gz -C ~/$v; local s=$(date +%s)
  cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/$v.cfg.log 2>&1
  cmake --build ~/$v/build -j 4 > $L/$v.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/$v.log; }
n=0
for v in "$@"; do b $v & n=$((n + 1)); [ $((n % 2)) = 0 ] && wait; done
wait
for v in "$@"; do echo "$v $(cat ~/$v/COMMIT | cut -c1-7) $(tail -n 1 $L/$v.log)"; done > ~/results/bisect_builds.txt
echo BUILD_DONE >> ~/results/bisect_builds.txt
