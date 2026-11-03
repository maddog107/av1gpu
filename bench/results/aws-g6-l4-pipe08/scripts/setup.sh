#!/bin/bash
# pipe08 instance setup: deps; av1gpu OLD (main cde1f74, tests OFF) and NEW (main + working changes, tests ON, k_recon
# objects linked from OLD: AV1GPU_RECON_OBJECTS) for sm_89; the v0.8.0 Linux release binary (~/rel); size-range photos
# (~/images), standard set (~/images_std); lossless sets (~/ll), identity sets (~/sets), mono set; palette sets
# (~/pal/sets).  Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/new ~/old ~/data ~/pal ~/rel
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick strace ltrace > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/old; tar -xzf ~/main.tar.gz -C ~/new; tar -xzf ~/data.tar.gz -C ~/data
echo cde1f74 > ~/old/COMMIT; echo work > ~/new/COMMIT
( cd ~/rel && curl -sSLO https://github.com/maddog107/av1gpu/releases/download/v0.8.0/av1gpu-0.8.0-linux-x86_64.tar.gz &&
  curl -sSLO https://github.com/maddog107/av1gpu/releases/download/v0.8.0/av1gpu-0.8.0-linux-x86_64.tar.gz.sha256 &&
  sha256sum -c av1gpu-0.8.0-linux-x86_64.tar.gz.sha256 && tar -xzf av1gpu-0.8.0-linux-x86_64.tar.gz ) > $L/rel.log 2>&1 &
( cd ~/old && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
FP=$!
( s=$(date +%s)
  cmake -S ~/old -B ~/old/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/old.cfg.log 2>&1
  cmake --build ~/old/build -j 16 > $L/old.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/old.log
  s=$(date +%s)
  cmake -S ~/new -B ~/new/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=ON \
    -DAV1GPU_RECON_OBJECTS=$HOME/old/build/CMakeFiles/av1gpu_recon.dir/encoder/src > $L/new.cfg.log 2>&1
  cmake --build ~/new/build -j 16 > $L/new.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/new.log
  echo BUILD_DONE >> $L/new.log ) &
BP=$!
wait $FP
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
( bash ~/mk_mono.sh > $L/mono.log 2>&1; echo "exit $?" >> $L/mono.log )
( cd ~ && python3 ~/prep_sets.py ~/old/bench/palette_study_images.csv ~/old/bench/openimages_v4_gamut.csv ~/old ~/render ~/pal \
    > $L/pal.log 2>&1; echo "exit $?" >> $L/pal.log ) &
wait
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  dav1d --version; convert -version | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  for d in ~/sets/*/ ~/pal/sets/*/; do echo "$d: $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log $L/sets.log $L/mono.log $L/pal.log $L/rel.log
echo SETUP_DONE
