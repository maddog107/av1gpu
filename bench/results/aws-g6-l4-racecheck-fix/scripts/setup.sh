#!/bin/bash
# k_recon racecheck fix check, instance setup (after bench/results/aws-g6-l4-speed-v04/scripts/setup.sh and -v06's):
# deps; av1gpu NEW (fix/recon-racecheck 78e374c, tests ON) and OLD (main = v0.6.0 d7d94b3) for sm_89; the size-range
# photos (~/images, 100) and standard set (~/images_std); the lossless sets (~/ll, mk_ll_sets.py, for Kodak); the
# RGBA set (~/data); the identity sets (~/sets, mk_sets.py, for the thumbnails).  Logs in ~/results/setup.
# Prints SETUP_DONE.
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
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  /usr/local/cuda/bin/compute-sanitizer --version | tail -1
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2; convert -version | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "new $(cat ~/new/COMMIT) old $(cat ~/old/COMMIT)"
  echo "images: $(ls ~/images | wc -l), images_std: $(ls ~/images_std | wc -l), kodak: $(ls ~/ll/kodak | wc -l), rgba: $(ls ~/data/rgba/set | wc -l)"
  for d in ~/sets/*/; do echo "sets/$(basename $d): $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log $L/sets.log
echo SETUP_DONE
