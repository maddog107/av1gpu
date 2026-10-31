#!/bin/bash
# Stage 1: the stock build, with the same commands as the published c6a SVT-AV1 run
# (libaom 3.13.1, SVT-AV1 v4.2.0, libavif 1.4.0, ssimulacra2 from libjxl v0.11.1; CMake Release, distro GCC),
# into ~/opt.  The image sets download in the background meanwhile.
set -ex
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil nasm yasm git build-essential libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev >/dev/null
( cd ~/av1gpu && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out bench/images_gamut > ~/fetch_gamut.log 2>&1
  python3 bench/fetch_images.py --out bench/images > ~/fetch_std.log 2>&1; echo FETCH_DONE >> ~/fetch_std.log ) &
mkdir -p ~/src ~/opt && cd ~/src
git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 >/dev/null
ninja -C aom-build install >/dev/null
git clone -q --depth 1 -b v4.2.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git svt
cmake -S svt -B svt-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF >/dev/null
ninja -C svt-build install >/dev/null
git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
PKG_CONFIG_PATH=$HOME/opt/lib/pkgconfig cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt \
  -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_SVT=SYSTEM -DAVIF_CODEC_DAV1D=OFF -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM \
  -DCMAKE_PREFIX_PATH=$HOME/opt -DCMAKE_INSTALL_RPATH=$HOME/opt/lib >/dev/null
ninja -C avif-build install >/dev/null
git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
  -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
  -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF >/dev/null
ninja -C jxl-build ssimulacra2 >/dev/null
cp jxl-build/tools/ssimulacra2 ~/opt/bin/
~/opt/bin/avifenc --version | head -3
wait
tail -2 ~/fetch_gamut.log ~/fetch_std.log
echo SETUP_DONE
