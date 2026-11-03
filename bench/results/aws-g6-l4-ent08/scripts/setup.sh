#!/bin/bash
# ent08 instance setup (g6.4xlarge, DLAMI base-oss-nvidia-driver-gpu-ubuntu-24.04, CUDA 13.2):
#  - main cde1f74 (~/main.tar.gz) -> ~/main, built for sm_89 with tests ON (ccache)
#  - ~/br: the branch tree (synced later with rsync), configured with AV1GPU_RECON_OBJECTS = main's k_recon objects
#    (this branch does not touch k_recon*)
#  - sets: size range (~/images), standard (~/images_std), ~/ll (mk_ll_sets.py), ~/data (RGBA), ~/sets (mk_sets.py),
#    palette sets (~/pal/sets: prep_sets.py of the palette study)
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/main ~/data ~/pal
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick ccache rsync > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/main; echo cde1f74 > ~/main/COMMIT; tar -xzf ~/data.tar.gz -C ~/data
( cd ~/main && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log
  ( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
  ( cd ~ && python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
  ( python3 ~/prep_sets.py ~/main/bench/palette_study_images.csv ~/main/bench/openimages_v4_gamut.csv ~/main ~/render ~/pal \
      > $L/palsets.log 2>&1; echo "exit $?" >> $L/palsets.log ) ) &
( s=$(date +%s)
  cmake -S ~/main -B ~/main/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DAV1GPU_TESTS=ON > $L/main.cfg.log 2>&1
  cmake --build ~/main/build -j 16 > $L/main.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/main.log ) &
wait
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  ls /usr/local/cuda/bin/ | grep -E 'sanitizer|nsys|ncu'
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  for d in ~/sets/*/ ~/pal/sets/*/; do echo "$d: $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/main.log $L/fetch.log $L/llsets.log $L/sets.log $L/palsets.log
echo SETUP_DONE
