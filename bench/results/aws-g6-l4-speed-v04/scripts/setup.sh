#!/bin/bash
# Speed rerun instance setup: deps; av1gpu NEW (perf/cpu-threads fdb0628) and OLD (v0.3.0 f8f1ecc) for sm_89; the
# size-range photos (~/images, 100) and standard set (~/images_std); the lossless comparison set (~/ll/cmp, from the
# lossless run's source tree in ~/br); the RGBA set (~/data); the demo folder (~/av1gpu-readme, 99 photos).
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/new ~/old ~/br ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential > $L/apt.log 2>&1
tar -xzf ~/new.tar.gz -C ~/new; tar -xzf ~/old.tar.gz -C ~/old; tar -xzf ~/br.tar.gz -C ~/br; tar -xzf ~/data.tar.gz -C ~/data
( cd ~/new && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
for v in new old; do
  ( cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 4 > $L/$v.log 2>&1; echo "exit $?" >> $L/$v.log ) &
done
wait
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
mkdir -p ~/av1gpu-readme/photos ~/av1gpu-readme/av1gpu
for f in ~/images/*.jpg; do [ "$(basename $f)" = 9fba6f263f3c4f3e_tiny.jpg ] || cp $f ~/av1gpu-readme/photos/; done
ln -sfn ~/new/build ~/av1gpu-readme/av1gpu/build
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "new $(cat ~/new/COMMIT) old $(cat ~/old/COMMIT)"
  echo "ll/cmp: $(ls ~/ll/cmp | wc -l) files, $(cat ~/ll/cmp/*.png | wc -c) bytes"
  echo "demo photos: $(ls ~/av1gpu-readme/photos | wc -l), $(du -sh ~/av1gpu-readme/photos | cut -f1)"; } > ~/results/env.txt 2>&1
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log
echo SETUP_DONE
