#!/bin/bash
# instance: deps, clone, images, data, build instr-main -> av1gpu-main and instr-new -> av1gpu-new
set -x
export DEBIAN_FRONTEND=noninteractive
mkdir -p ~/results ~/tmp ~/bin ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev python3-numpy python3-pil build-essential >/dev/null
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; /usr/local/cuda/bin/nvcc --version | tail -1; nproc;
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'; lscpu -e=CPU,CORE; } > ~/results/env.txt 2>&1
git clone -q --no-checkout ~/src2.bundle ~/av1gpu
cd ~/av1gpu && git fetch -q ~/src2.bundle 'refs/heads/*:refs/heads/*' && git checkout -q instr-main
python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images_gamut > ~/results/fetch.log 2>&1
cd ~ && python3 mkdata.py > ~/results/mkdata.log 2>&1
cd ~/av1gpu
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > ~/results/cfg.log 2>&1
for b in instr-main instr-new; do
  git checkout -q $b; t0=$(date +%s); ninja -C build > ~/results/build_$b.log 2>&1; rc=$?
  echo "$b exit $rc $(( $(date +%s) - t0 )) s $(git log --oneline -1)"
  [ $rc = 0 ] && cp build/av1gpu ~/bin/av1gpu-${b#instr-}
done
echo SETUP2_DONE
