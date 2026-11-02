#!/bin/bash
# crop16 instance setup (after bench/results/aws-g6-l4-speed-v04/scripts/setup.sh): deps; av1gpu MAIN (d7d94b3, tests
# OFF) and the branch CR (~/cr, tests ON) for sm_89; the size-range photos (~/images, 100) and standard set
# (~/images_std); the lossless sets (~/ll, mk_ll_sets.py: kodak, ...); the RGBA set (~/data); the identity sets
# (~/sets, mk_sets.py: thumbnails, odd sizes, gray + alpha, mixed list, WebP, single); ImageMagick 7 (7.1.2-35 AppImage, ~/im7)
# if it runs here, else ImageMagick 6 (convert).  Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/main ~/cr ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools imagemagick libimage-exiftool-perl time > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/main; echo d7d94b3 > ~/main/COMMIT
tar -xzf ~/cr.tar.gz -C ~/cr; cat ~/cr.commit > ~/cr/COMMIT
tar -xzf ~/data.tar.gz -C ~/data; ln -sfn ~/main ~/br
( cd ~/main && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "$(ls ~/images | wc -l) $(ls ~/images_std | wc -l)" >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
for v in main cr; do
  t=OFF; [ $v = cr ] && t=ON
  ( s=$(date +%s)
    cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=$t > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 4 > $L/$v.log 2>&1; echo "exit $? after $(( $(date +%s) - s )) s" >> $L/$v.log ) &
done
( cd ~ && curl -fsSL -o ~/magick https://github.com/ImageMagick/ImageMagick/releases/download/7.1.2-35/ImageMagick-7.1.2-35-gcc-x86_64.AppImage \
    > $L/im7.log 2>&1 && chmod +x ~/magick && ./magick --appimage-extract > /dev/null 2>&1 && mv squashfs-root im7 &&
  ~/im7/AppRun -version >> $L/im7.log 2>&1; echo "im7 exit $?" >> $L/im7.log ) &
wait
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  ls /usr/local/cuda/bin/ | grep -E 'sanitizer|nsys|ncu'
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  dav1d --version; aomdec --help 2>&1 | grep -i 'av1 decoder' | head -2; convert -version | head -1
  tail -3 $L/im7.log
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "main $(cat ~/main/COMMIT) cr $(cat ~/cr/COMMIT)"
  for d in ~/sets/*/; do echo "sets/$(basename $d): $(ls $d | wc -l) files"; done
  echo "kodak: $(ls ~/ll/kodak | wc -l), rgba: $(ls ~/data/rgba/set | wc -l)"; } > ~/results/env.txt 2>&1
tail -n 2 $L/main.log $L/cr.log $L/fetch.log $L/llsets.log $L/sets.log $L/im7.log
echo SETUP_DONE
