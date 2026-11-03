#!/bin/bash
# an08 instance setup: deps; av1gpu OLD (main cde1f74, tests OFF) and NEW (working tree, tests ON) for sm_89;
# size-range photos (~/images), standard set (~/images_std); lossless sets (~/ll), identity sets (~/sets),
# mono set; palette sets (~/pal/sets: screen, flat_png, flat_jpg ...).  Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/new ~/old ~/data ~/pal
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/old; tar -xzf ~/main.tar.gz -C ~/new; tar -xzf ~/data.tar.gz -C ~/data
echo cde1f74 > ~/old/COMMIT; echo work > ~/new/COMMIT
( cd ~/old && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
FP=$!
for v in old new; do
  t=OFF; [ $v = new ] && t=ON
  ( s=$(date +%s)
    cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=$t > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 16 > $L/$v.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/$v.log ) &
done
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
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log $L/sets.log $L/mono.log $L/pal.log
echo SETUP_DONE
