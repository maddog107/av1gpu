#!/bin/bash
# Loop-restoration study instance setup (c7a.16xlarge, 64 vCPU, Canonical Ubuntu 24.04; no GPU, no av1gpu build).
#  - libaom v3.13.1 (shared, encoder + decoder) + libavif v1.4.0 (aom + system dav1d) -> ~/opt/bin/{avifenc,avifdec}:
#    the versions of the stored baselines (bench/results/aws-g6-l4-palette-study/scripts/setup.sh)
#  - libjxl v0.11.1 ssimulacra2 and butteraugli_main -> ~/opt/bin (in the run, butteraugli_main was built later
#    in the same tree with `ninja -C jxl-build butteraugli_main`)
#  - the libaom source stays in ~/src/aom for reading av1/encoder/speed_features.c; the inspection decoder
#    (lrstat.c) and the patched libaom are built by build_lr.sh
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/src ~/opt/bin
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev python3-numpy python3-pil \
  nasm git build-essential libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev dav1d \
  xz-utils time > $L/apt.log 2>&1
( cd ~/src && git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
  cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF > $L/jxl.cfg.log 2>&1
  ninja -C jxl-build -j 24 ssimulacra2 butteraugli_main > $L/jxl.log 2>&1; echo "exit $?" >> $L/jxl.log
  cp jxl-build/tools/ssimulacra2 jxl-build/tools/butteraugli_main ~/opt/bin/ ) &
( set -e; cd ~/src
  git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
  cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DENABLE_TESTS=0 -DENABLE_DOCS=0 \
    -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 > $L/aom.log 2>&1
  ninja -C aom-build -j 32 install >> $L/aom.log 2>&1
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  PKG_CONFIG_PATH=$HOME/opt/lib/pkgconfig cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/opt -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM \
    -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_PREFIX_PATH=$HOME/opt \
    -DCMAKE_INSTALL_RPATH=$HOME/opt/lib > $L/avif.log 2>&1
  ninja -C avif-build install >> $L/avif.log 2>&1; echo "exit 0" >> $L/avif.log ) &
wait
{ ~/opt/bin/avifenc --version 2>&1 | head -3; ~/opt/bin/avifdec --version 2>&1 | head -3; dav1d --version
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
} > ~/results/env.txt 2>&1
tail -n 2 $L/jxl.log $L/aom.log $L/avif.log
ls -la ~/opt/bin
echo SETUP_DONE
