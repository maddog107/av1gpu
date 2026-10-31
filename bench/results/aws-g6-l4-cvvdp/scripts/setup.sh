#!/bin/bash
# On the instance: deps; libaom 3.13.1, SVT-AV1 4.2.0, libavif 1.4.0 (+dav1d decoder) and libjxl 0.11.1
# ssimulacra2 built as in scratchpad/svt/setup.sh; av1gpu (bench/cvvdp archive); the gamut images; ffmpeg n8.1
# (BtbN static) for NVENC; a venv with cvvdp + CUDA torch.  Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/src ~/opt ~/av1gpu
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  python3-venv nasm yasm git build-essential libhwy-dev libbrotli-dev liblcms2-dev libgif-dev libyuv-dev libdav1d-dev xz-utils \
  > $L/apt.log 2>&1
tar -xzf ~/av1gpu.tar.gz -C ~/av1gpu
# images (the manifest's sha256 is checked by fetch_images.py); keep the 99 photos of the published SVT/NVENC runs
( cd ~/av1gpu && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images_all > $L/fetch.log 2>&1
  mkdir -p ~/images_gamut && for f in $(cat ~/photos99.txt); do ln -f ~/images_all/$f ~/images_gamut/$f; done
  ls ~/images_gamut | wc -l >> $L/fetch.log; echo FETCH_DONE >> $L/fetch.log ) &
# venv with cvvdp (pulls torch with its own CUDA runtime)
( python3 -m venv ~/venv && ~/venv/bin/pip install -q --upgrade pip && ~/venv/bin/pip install cvvdp > $L/pip.log 2>&1
  echo "pip exit $?" >> $L/pip.log ) &
# ffmpeg n8.1 static (the published NVENC run used n8.1.3 on driver 595.112.07)
( mkdir -p ~/f81 && curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz \
  | tar xJ -C ~/f81 --strip-components=1 && sudo cp ~/f81/bin/ffmpeg ~/f81/bin/ffprobe /usr/local/bin/ \
  && /usr/local/bin/ffmpeg -version | head -1 > $L/ffmpeg.log 2>&1 ) &
# av1gpu
( cmake -S ~/av1gpu -B ~/av1gpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > $L/av1gpu.cfg.log 2>&1
  cmake --build ~/av1gpu/build -j 6 > $L/av1gpu.log 2>&1; echo "exit $?" >> $L/av1gpu.log ) &
# ssimulacra2
( cd ~/src && git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl
  cmake -S libjxl -B jxl-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF > $L/jxl.cfg.log 2>&1
  ninja -C jxl-build -j 4 ssimulacra2 > $L/jxl.log 2>&1; echo "exit $?" >> $L/jxl.log ) &
# libaom -> SVT-AV1 -> libavif
( set -e; cd ~/src
  git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
  cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DENABLE_TESTS=0 -DENABLE_DOCS=0 \
    -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 > $L/aom.log 2>&1
  ninja -C aom-build -j 6 install >> $L/aom.log 2>&1
  git clone -q --depth 1 -b v4.2.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git svt
  cmake -S svt -B svt-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF > $L/svt.log 2>&1
  ninja -C svt-build -j 6 install >> $L/svt.log 2>&1
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  PKG_CONFIG_PATH=$HOME/opt/lib/pkgconfig cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt \
    -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_SVT=SYSTEM -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM \
    -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_PREFIX_PATH=$HOME/opt -DCMAKE_INSTALL_RPATH=$HOME/opt/lib \
    > $L/avif.log 2>&1
  ninja -C avif-build install >> $L/avif.log 2>&1; echo "exit 0" >> $L/avif.log ) &
wait
cp ~/src/jxl-build/tools/ssimulacra2 ~/opt/bin/
printf '#!/bin/sh\nexport GP_ENCODERS=4\nexec %s "$@"\n' ~/av1gpu/build/av1gpu > ~/w4.sh; chmod +x ~/w4.sh
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader; /usr/local/cuda/bin/nvcc --version | tail -2
  ~/opt/bin/avifenc --version | head -3; ~/opt/bin/avifdec --version | head -3; cwebp -version; dwebp -version
  cat $L/ffmpeg.log; ~/venv/bin/pip list 2>/dev/null | grep -i -E "^(cvvdp|torch|torchvision|numpy|pillow|scipy) "
  ~/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  nproc; free -g | head -2; df -h / | tail -1; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  echo "av1gpu commit $(cat ~/COMMIT)"; } > ~/results/env.txt 2>&1
tail -2 $L/av1gpu.log $L/jxl.log $L/avif.log $L/pip.log $L/fetch.log
ls -la ~/opt/bin ~/av1gpu/build/av1gpu
echo SETUP_DONE
