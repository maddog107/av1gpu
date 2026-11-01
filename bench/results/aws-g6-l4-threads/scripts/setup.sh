#!/bin/bash
# On the instance: deps, clone the bundle (~/av1gpu), fetch the size-range images.  Logs in ~/results.
set -x
export DEBIAN_FRONTEND=noninteractive
mkdir -p ~/results ~/tmp ~/bin ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev python3-numpy python3-pil \
  build-essential time >/dev/null
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; /usr/local/cuda/bin/nvcc --version | tail -2;
  cmake --version | head -1; gcc --version | head -1; nproc; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'; free -g | head -2;
  cat /sys/fs/cgroup/cgroup.controllers; cat /proc/self/cgroup; } > ~/results/env.txt 2>&1
git clone -q --no-checkout ~/src.bundle ~/av1gpu
cd ~/av1gpu && git fetch -q ~/src.bundle 'refs/remotes/origin/main:refs/heads/main' 'refs/heads/perf/cpu-threads:refs/heads/perf/cpu-threads' \
  'refs/remotes/origin/contrib/batch-fallback:refs/heads/contrib/batch-fallback' && git checkout -q main
git -C ~/av1gpu branch -a > ~/results/branches.txt
python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images_gamut > ~/results/fetch_gamut.log 2>&1
echo SETUP_DONE
