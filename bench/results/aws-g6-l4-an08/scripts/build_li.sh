#!/bin/bash
# build_li.sh: ~/new with -lineinfo (profiling only) in ~/new/build_li -> ~/results/build_li.log
s=$(date +%s)
cmake -S ~/new -B ~/new/build_li -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF -DCMAKE_CUDA_FLAGS=-lineinfo > ~/results/build_li.log 2>&1
cmake --build ~/new/build_li -j 24 --target av1gpu >> ~/results/build_li.log 2>&1
echo "exit $? after $(( $(date +%s) - s )) s" >> ~/results/build_li.log
