#!/bin/bash
# Palette study instance setup (g6.4xlarge, Deep Learning Base OSS NVIDIA Driver AMI, Ubuntu 24.04).
#  - av1gpu main (60f94ff, ~/main.tar.gz) for sm_89 -> ~/av1gpu/build/av1gpu
#  - libaom v3.13.1 (shared, encoder + decoder) + libavif v1.4.0 (aom + system dav1d) -> ~/opt/bin/{avifenc,avifdec}:
#    the versions of the stored baselines (bench/results/aws-g6-l4-tiles4x4/scripts/aom.sh)
#  - libaom v3.13.1 again, decoder only with CONFIG_INSPECTION=1, examples/inspect.c replaced by ~/pstat.c ->
#    ~/opt/bin/pstat (palette / IntraBC block counts and frame-header flags per AV1 frame)
#  - libjxl v0.11.1 ssimulacra2 and butteraugli_main -> ~/opt/bin
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/src ~/opt/bin ~/av1gpu
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  nasm git build-essential libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev dav1d \
  librsvg2-bin fonts-dejavu-core xz-utils time > $L/apt.log 2>&1
tar -xzf ~/main.tar.gz -C ~/av1gpu
( cmake -S ~/av1gpu -B ~/av1gpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/av1gpu.cfg.log 2>&1
  cmake --build ~/av1gpu/build -j 8 > $L/av1gpu.log 2>&1; echo "exit $?" >> $L/av1gpu.log ) &
( cd ~/src && git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
  cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF > $L/jxl.cfg.log 2>&1
  ninja -C jxl-build -j 4 ssimulacra2 butteraugli_main > $L/jxl.log 2>&1; echo "exit $?" >> $L/jxl.log
  cp jxl-build/tools/ssimulacra2 jxl-build/tools/butteraugli_main ~/opt/bin/ ) &
( set -e; cd ~/src
  git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
  cp -r aom aom-insp; cp ~/pstat.c aom-insp/examples/inspect.c
  ( cmake -S aom-insp -B aom-insp-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONFIG_AV1_ENCODER=0 -DCONFIG_INSPECTION=1 \
      -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=0 -DBUILD_SHARED_LIBS=0 > $L/aom-insp.log 2>&1
    ninja -C aom-insp-build -j 4 inspect >> $L/aom-insp.log 2>&1; echo "exit $?" >> $L/aom-insp.log
    cp aom-insp-build/examples/inspect ~/opt/bin/pstat || cp aom-insp-build/inspect ~/opt/bin/pstat ) &
  cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DENABLE_TESTS=0 -DENABLE_DOCS=0 \
    -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 > $L/aom.log 2>&1
  ninja -C aom-build -j 8 install >> $L/aom.log 2>&1
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  PKG_CONFIG_PATH=$HOME/opt/lib/pkgconfig cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/opt -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM \
    -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_PREFIX_PATH=$HOME/opt \
    -DCMAKE_INSTALL_RPATH=$HOME/opt/lib > $L/avif.log 2>&1
  ninja -C avif-build install >> $L/avif.log 2>&1; echo "exit 0" >> $L/avif.log
  wait ) &
wait
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -2
  ~/opt/bin/avifenc --version 2>&1 | head -3; ~/opt/bin/avifdec --version 2>&1 | head -3; dav1d --version; cwebp -version
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  rsvg-convert --version
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "av1gpu $(cat ~/av1gpu/COMMIT 2>/dev/null)"; } > ~/results/env.txt 2>&1
tail -n 2 $L/av1gpu.log $L/jxl.log $L/aom.log $L/aom-insp.log $L/avif.log
ls -la ~/opt/bin ~/av1gpu/build/av1gpu
echo SETUP_DONE
