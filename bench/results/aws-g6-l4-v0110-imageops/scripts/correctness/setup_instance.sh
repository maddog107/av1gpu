#!/bin/bash
# instance setup: main + PR side by side (sm_89, tests ON), ImageMagick 7 from source, image sets
set -x
export DEBIAN_FRONTEND=noninteractive
sudo shutdown -c; sudo shutdown -h +480
L=~/results/setup; mkdir -p $L ~/main ~/pr ~/images ~/images_std
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools ccache rsync liblcms2-dev libtiff-dev zlib1g-dev git \
  libjpeg-turbo-progs libimage-exiftool-perl libavif-bin > $L/apt.log 2>&1
echo apt exit $?
tar -xzf ~/main.tar.gz -C ~/main; echo 4721191 > ~/main/COMMIT
tar -xzf ~/pr65.tar.gz -C ~/pr; echo c466e88 > ~/pr/COMMIT
( s=$(date +%s)
  cmake -S ~/main -B ~/main/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DAV1GPU_TESTS=ON > $L/main.cfg.log 2>&1
  cmake --build ~/main/build -j 16 > $L/main.log 2>&1; echo "main exit $? after $(( $(date +%s) - s )) s" >> $L/main.log
  # the PR tree: full build of its own, no shared objects, so nothing of main can leak into it
  s=$(date +%s)
  cmake -S ~/pr -B ~/pr/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DAV1GPU_TESTS=ON > $L/pr.cfg.log 2>&1
  cmake --build ~/pr/build -j 16 > $L/pr.log 2>&1; echo "pr exit $? after $(( $(date +%s) - s )) s" >> $L/pr.log ) &
( cd ~ && python3 ~/main/bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  cd ~/main && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log
  head -1 bench/palette_study_images.csv > ~/comics.csv; grep "^comics" bench/palette_study_images.csv >> ~/comics.csv
  python3 bench/fetch_images.py --manifest ~/comics.csv --out ~/comics > $L/comics.log 2>&1 ) &
( s=$(date +%s); cd ~ && git clone -q --depth 1 --branch 7.1.2-18 https://github.com/ImageMagick/ImageMagick.git im7 > $L/im.log 2>&1
  cd ~/im7 && ./configure --prefix=$HOME/im --with-quantum-depth=16 --enable-hdri --without-x --without-perl --disable-docs --disable-openmp >> $L/im.log 2>&1
  make -j 8 >> $L/im.log 2>&1 && make install >> $L/im.log 2>&1; echo "im exit $? after $(( $(date +%s) - s )) s" >> $L/im.log ) &
wait
python3 ~/t/gen.py > $L/gen.log 2>&1; python3 ~/t/genrand.py >> $L/gen.log 2>&1
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  ls /usr/local/cuda/bin/ | grep -E 'sanitizer|nsys|ncu'
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2
  ~/im/bin/magick -version | head -3
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  } > ~/results/env.txt 2>&1
tail -n 2 $L/main.log $L/pr.log $L/fetch.log $L/fetch_std.log $L/im.log
echo SETUP_DONE
