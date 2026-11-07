#!/bin/bash
# mkproto.sh NAME: ~/proto/NAME = the PR tree with ~/protosrc/NAME/* over encoder/src, built like ~/pr (same flags, main's k_recon objects)
n=$1; T=~/proto/$n; mkdir -p $T; rsync -a --delete --exclude build ~/pr/ $T/; cp ~/protosrc/$n/* $T/encoder/src/
cmake -S $T -B $T/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DAV1GPU_TESTS=OFF \
  -DAV1GPU_RECON_OBJECTS=$HOME/main/build/CMakeFiles/av1gpu_recon.dir/encoder/src > ~/results/proto_$n.cfg.log 2>&1
cmake --build $T/build -j 16 > ~/results/proto_$n.log 2>&1; echo "proto $n exit $?"; ls -l $T/build/av1gpu
