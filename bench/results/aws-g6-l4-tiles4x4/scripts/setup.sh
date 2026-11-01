#!/bin/bash
# On the instance: deps; av1gpu NEW (perf/tiles-4x4) and OLD (v0.3.0) for sm_89; libjxl 0.11.1 ssimulacra2;
# libavif 1.4.0 avifdec (system dav1d 1.4.1); venv with cvvdp 0.5.7 + CUDA torch (versions of the published CVVDP run);
# the size-range photos (~/images, 99) and the standard set (~/images_std, 100); the lossless and RGBA test sets.
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/src ~/opt/bin ~/new ~/old ~/data
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  python3-venv nasm git build-essential libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev dav1d aom-tools \
  librsvg2-bin fonts-dejavu-core xz-utils > $L/apt.log 2>&1
tar -xzf ~/new.tar.gz -C ~/new; tar -xzf ~/old.tar.gz -C ~/old; tar -xzf ~/data.tar.gz -C ~/data
ln -sfn ~/new ~/br
gcc -O2 -o ~/data/tools/pngraw ~/data/tools/pngraw.c -lpng > $L/pngraw.log 2>&1
( cd ~/new && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  ls ~/images | wc -l >> $L/fetch.log; ls ~/images_std | wc -l >> $L/fetch_std.log; echo FETCH_DONE >> $L/fetch.log ) &
( python3 -m venv ~/venv && ~/venv/bin/pip install -q --upgrade pip &&
  { ~/venv/bin/pip install cvvdp==0.5.7 torch==2.15.0 numpy==2.5.4 pillow==13.0.0 scipy==1.18.1 ||
    ~/venv/bin/pip install cvvdp==0.5.7; } > $L/pip.log 2>&1
  echo "pip exit $?" >> $L/pip.log ) &
for v in new old; do
  ( cmake -S ~/$v -B ~/$v/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/$v.cfg.log 2>&1
    cmake --build ~/$v/build -j 4 > $L/$v.log 2>&1; echo "exit $?" >> $L/$v.log ) &
done
( cd ~/src && git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
  cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF > $L/jxl.cfg.log 2>&1
  ninja -C jxl-build -j 4 ssimulacra2 > $L/jxl.log 2>&1; echo "exit $?" >> $L/jxl.log ) &
( set -e; cd ~/src
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt \
    -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM \
    -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_INSTALL_RPATH=$HOME/opt/lib > $L/avif.log 2>&1
  ninja -C avif-build install >> $L/avif.log 2>&1; echo "exit 0" >> $L/avif.log ) &
wait
cp ~/src/jxl-build/tools/ssimulacra2 ~/opt/bin/
# lossless sets (~/ll) need ~/br (the source tree) and ~/images
( cd ~ && python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -2
  ~/opt/bin/avifdec --version | head -1; dav1d --version; aomdec --help 2>&1 | grep -i "av1" | head -2
  ~/venv/bin/pip list 2>/dev/null | grep -i -E "^(cvvdp|torch|numpy|pillow|scipy) "
  ~/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python3 -c "import PIL, numpy; print('system pillow', PIL.__version__, 'numpy', numpy.__version__)"
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "new $(cat ~/new/COMMIT) old $(cat ~/old/COMMIT)"; } > ~/results/env.txt 2>&1
tail -2 $L/new.log $L/old.log $L/jxl.log $L/avif.log $L/pip.log $L/fetch.log $L/fetch_std.log $L/llsets.log
ls -la ~/opt/bin ~/new/build/av1gpu ~/old/build/av1gpu
echo SETUP_DONE
