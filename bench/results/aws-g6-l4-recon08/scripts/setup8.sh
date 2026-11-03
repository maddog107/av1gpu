#!/bin/bash
# recon08 instance setup (after speed-v06's setup.sh): deps + ccache; ~/old = main cde1f74 (tests OFF) and
# ~/new = same source to start with (tests ON), both sm_89; images; ll sets; RGBA (~/data); identity sets (~/sets);
# mono set; palette sets (~/pal/sets).  Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/new ~/old ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick ccache > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/old; tar -xzf ~/main.tar.gz -C ~/new; tar -xzf ~/data.tar.gz -C ~/data
cp ~/main.commit ~/old/COMMIT; cp ~/main.commit ~/new/COMMIT
( cd ~/new && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
FP=$!
for v in old new; do
  t=OFF; [ $v = new ] && t=ON
  ( s=$(date +%s)
    cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=$t -DAV1GPU_RECON_JOBS=3 \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 14 > $L/$v.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/$v.log ) &
done
wait $FP
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
( cd ~ && bash ~/mk_mono.sh > $L/mono.log 2>&1; echo "exit $?" >> $L/mono.log )
( mkdir -p ~/pal && cd ~/pal && python3 ~/prep_sets.py ~/new/bench/palette_study_images.csv ~/new/bench/openimages_v4_gamut.csv ~/new ~/render ~/pal > $L/palsets.log 2>&1; echo "exit $?" >> $L/palsets.log )
wait
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  /usr/local/cuda/bin/compute-sanitizer --version | tail -1
  ls /usr/local/cuda/bin/ | grep -E 'nsys|ncu'
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2; convert -version | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "new $(cat ~/new/COMMIT) old $(cat ~/old/COMMIT)"
  echo "images: $(ls ~/images | wc -l), images_std: $(ls ~/images_std | wc -l), kodak: $(ls ~/ll/kodak | wc -l), rgba: $(ls ~/data/rgba/set | wc -l)"
  for d in ~/sets/*/ ~/pal/sets/*/; do echo "$d: $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/new.log $L/old.log $L/fetch.log $L/llsets.log $L/sets.log $L/mono.log $L/palsets.log
echo SETUP_DONE
