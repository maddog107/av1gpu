#!/bin/bash
# l4e.sh: fetch the CUDA 13.4 release-style build from the Blackwell instance; on this L4 (driver 580.196.07): its
# start-up and v0.6.0's (installed UMD and the 13.4 compat UMD), cmp against v0.6.0 on the three sets, speed.  L4E_DONE.
mkdir -p ~/rb/main134 && scp -q -i ~/.ssh/xfer ubuntu@$BLACKWELL:rb/main134/av1gpu ~/rb/main134/av1gpu && echo "fetched $(sha256sum ~/rb/main134/av1gpu | cut -c1-16)"
N=~/rb/main134/av1gpu; O=~/rel/0.6.0/av1gpu; U=/usr/local/cuda-13.4/compat
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader
bash ~/startup.sh $N l4-580-main134 2
bash ~/startup.sh $O l4-580-v060 2
LD_LIBRARY_PATH=$U bash ~/startup.sh $N l4-580-umd134-main134 1
LD_LIBRARY_PATH=$U bash ~/startup.sh $O l4-580-umd134-v060 1
bash ~/cmpsets.sh $O $N l4-580-v060-vs-main134
bash ~/spdab.sh $O $N ~/sr3 2 CUDA_CACHE_DISABLE=1 | tail -2
echo L4E_DONE
