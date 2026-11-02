#!/bin/bash
# bwfinal2.sh: the rest of bwfinal.sh after the forced PTX JIT (stopped after 10 min): cmp and speed.  BWFINAL2_DONE.
N=~/rb/main134/av1gpu; O=~/rel/0.6.0/av1gpu; U=/usr/local/cuda-13.4/compat
echo "== cmp (595)"
bash ~/cmpsets.sh $O $N v060-vs-main134
echo "== cmp (13.4 UMD)"
bash ~/cmpsets.sh $O $N v060-vs-main134-umd134 LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$HOME/spd/cacheB
echo "== speed 595: a = v0.6.0, b = CUDA 13.4 build"
bash ~/spdab.sh $O $N ~/sr3 2 CUDA_CACHE_DISABLE=1 | tail -2
echo "== speed 13.4 UMD: a = v0.6.0 (re-finalized, warm cache), b = CUDA 13.4 build"
bash ~/spdab.sh $O $N ~/sr3 2 LD_LIBRARY_PATH=$U CUDA_CACHE_PATH=$HOME/spd/cacheB | tail -2
echo BWFINAL2_DONE
