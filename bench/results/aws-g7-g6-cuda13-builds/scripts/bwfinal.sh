#!/bin/bash
# bwfinal.sh: the CUDA 13.4 release-style build (~/rb/main134/av1gpu) against v0.6.0 on the Blackwell: start-up
# (595 and 615.71.10 UMDs), PTX-only start-up (CUDA_FORCE_PTX_JIT=1, what a GPU without SASS gets), outputs (cmp,
# three sets) and speed (size range x3; 615: v0.6.0 with its re-finalized kernels in a warm cache).  BWFINAL_DONE.
N=~/rb/main134/av1gpu; O=~/rel/0.6.0/av1gpu; U=/usr/local/cuda-13.4/compat
echo "== start-up"
bash ~/startup.sh $N main134 2
LD_LIBRARY_PATH=$U bash ~/startup.sh $N main134-umd13.4 2
LD_LIBRARY_PATH=$U bash ~/startup.sh ~/rel/0.5.0/av1gpu v050-umd13.4 1
bash ~/startup.sh ~/rel/0.5.0/av1gpu v050 1
echo "== PTX JIT forced (595 UMD, then 13.4 UMD)"
for L in "" $U; do
  rm -rf ~/ptxc; s=$(date +%s.%N)
  env LD_LIBRARY_PATH=$L CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_PATH=$HOME/ptxc $N --batch ~/st/main134/list.txt --outdir /tmp/ptxo > /tmp/ptx.log 2>&1
  echo "forced PTX JIT UMD ${L:-595}: rc=$? $(printf %.2f $(echo "$(date +%s.%N) - $s" | bc)) s, cache $(du -sm ~/ptxc | cut -f1) MB"
  s=$(date +%s.%N); env LD_LIBRARY_PATH=$L CUDA_FORCE_PTX_JIT=1 CUDA_CACHE_PATH=$HOME/ptxc $N --batch ~/st/main134/list.txt --outdir /tmp/ptxo > /tmp/ptx.log 2>&1
  echo "forced PTX JIT UMD ${L:-595}, warm: rc=$? $(printf %.2f $(echo "$(date +%s.%N) - $s" | bc)) s"
done
echo "== cmp (595)"
bash ~/cmpsets.sh $O $N v060-vs-main134
echo "== cmp (13.4 UMD)"
bash ~/cmpsets.sh $O $N v060-vs-main134-umd134 LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$HOME/spd/cacheB
echo "== speed 595: a = v0.6.0, b = CUDA 13.4 build"
bash ~/spdab.sh $O $N ~/sr3 2 CUDA_CACHE_DISABLE=1 | tail -2
echo "== speed 13.4 UMD: a = v0.6.0 (re-finalized, warm cache), b = CUDA 13.4 build"
bash ~/spdab.sh $O $N ~/sr3 2 LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$HOME/spd/cacheB | tail -2
echo BWFINAL_DONE
