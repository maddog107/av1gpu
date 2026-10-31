#!/bin/bash
# On the instance: deps, both builds (main 7edea45 = ~/main, feat/444-10bit = ~/br), image sets, and the CPU
# codecs as published (svt/setup.sh: libaom 3.13.1, SVT-AV1 4.2.0, libavif 1.4.0, libjxl 0.11.1 ssimulacra2;
# libavif additionally with the system dav1d for avifdec -c dav1d).  Logs in ~/results.
set -x
export DEBIAN_FRONTEND=noninteractive
mkdir -p ~/results ~/tmp
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  dav1d aom-tools build-essential git nasm yasm libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev >/dev/null
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader; ls -d /usr/local/cuda*;
  /usr/local/cuda/bin/nvcc --version | tail -2; gcc --version | head -1; cmake --version | head -1; dav1d --version;
  aomdec --help 2>&1 | grep -i "av1 decoder" | head -2; uname -r; grep PRETTY /etc/os-release; nproc;
  grep -m1 "model name" /proc/cpuinfo; free -g | head -2; } > ~/results/env.txt 2>&1
mkdir -p ~/br ~/main
tar -xzf ~/br.tar.gz -C ~/br
tar -xzf ~/main.tar.gz -C ~/main
# image sets in the background
( cd ~/br && python3 bench/fetch_images.py --out ~/images > ~/results/fetch_std.log 2>&1
  python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images_gamut > ~/results/fetch_gamut.log 2>&1
  echo FETCH_DONE >> ~/results/fetch_std.log ) &
build() {   # tree builddir jobs [cmake options]
  cmake -S ~/$1 -B ~/$2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc $4 > ~/results/$2.cfg.log 2>&1
  cmake --build ~/$2 -j $3 > ~/results/$2.log 2>&1
  echo "exit $?" >> ~/results/$2.log
}
cpu_codecs() {
  set -e
  mkdir -p ~/src ~/opt && cd ~/src
  git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
  cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 >/dev/null
  nice ninja -C aom-build -j 6 install >/dev/null
  git clone -q --depth 1 -b v4.2.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git svt
  cmake -S svt -B svt-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF >/dev/null
  nice ninja -C svt-build -j 6 install >/dev/null
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  PKG_CONFIG_PATH=$HOME/opt/lib/pkgconfig cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt \
    -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_SVT=SYSTEM -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM \
    -DCMAKE_PREFIX_PATH=$HOME/opt -DCMAKE_INSTALL_RPATH=$HOME/opt/lib >/dev/null
  nice ninja -C avif-build -j 6 install >/dev/null
  git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
  cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF >/dev/null
  nice ninja -C jxl-build -j 6 ssimulacra2 >/dev/null
  cp jxl-build/tools/ssimulacra2 ~/opt/bin/
  { ~/opt/bin/avifenc --version; ~/opt/bin/avifdec --version; ~/opt/bin/ssimulacra2 2>&1 | head -2; cwebp -version; } 2>&1
  echo CODECS_DONE
}
( cpu_codecs > ~/results/codecs.log 2>&1 ) &
build br build-br 6 &
build main build-main 6 -DAV1GPU_TESTS=OFF &
wait
tail -3 ~/results/build-br.log ~/results/build-main.log ~/results/fetch_std.log ~/results/fetch_gamut.log ~/results/codecs.log
echo SETUP_DONE
