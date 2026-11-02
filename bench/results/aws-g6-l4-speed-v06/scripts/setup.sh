#!/bin/bash
# v0.6.0 release-check instance setup: deps; av1gpu NEW (main ef2522f, tests ON) and OLD (v0.5.0 b6545c1) for sm_89;
# the size-range photos (~/images, 100) and standard set (~/images_std); the lossless sets (~/ll, mk_ll_sets.py:
# cmp, kodak, gray, misc, webp, jpeg); the RGBA set (~/data); the identity sets (~/sets, mk_sets.py: thumbnails,
# odd sizes, gray + alpha, mixed list, WebP); the demo folder (~/av1gpu-readme, 99 photos).
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/new ~/old ~/br ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick > $L/apt.log 2>&1
tar -xzf ~/new.tar.gz -C ~/new; tar -xzf ~/old.tar.gz -C ~/old; tar -xzf ~/br.tar.gz -C ~/br; tar -xzf ~/data.tar.gz -C ~/data
( cd ~/new && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
for v in new old; do
  t=OFF; [ $v = new ] && t=ON
  ( s=$(date +%s)
    cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=$t > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 4 > $L/$v.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/$v.log ) &
done
wait
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
mkdir -p ~/av1gpu-readme/photos ~/av1gpu-readme/av1gpu
for f in ~/images/*.jpg; do [ "$(basename $f)" = 9fba6f263f3c4f3e_tiny.jpg ] || cp $f ~/av1gpu-readme/photos/; done
ln -sfn ~/new/build ~/av1gpu-readme/av1gpu/build
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  ls /usr/local/cuda/bin/ | grep -E 'sanitizer|nsys|ncu'
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2; convert -version | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "new $(cat ~/new/COMMIT) old $(cat ~/old/COMMIT)"
  echo "ll/cmp: $(ls ~/ll/cmp | wc -l) files, $(cat ~/ll/cmp/*.png | wc -c) bytes"
  for d in ~/sets/*/; do echo "sets/$(basename $d): $(ls $d | wc -l) files"; done
  echo "demo photos: $(ls ~/av1gpu-readme/photos | wc -l), $(du -sh ~/av1gpu-readme/photos | cut -f1)"; } > ~/results/env.txt 2>&1
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log $L/sets.log
echo SETUP_DONE
