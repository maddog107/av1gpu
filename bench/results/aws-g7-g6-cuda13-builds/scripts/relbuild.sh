#!/bin/bash
# relbuild.sh TAR NAME [IMAGE]: release-style Linux build of TAR (git archive) as release.yml does it (nvidia/cuda:12.9.1-devel-
# ubuntu22.04, default architectures, static deps, strip --strip-debug, ldd check), in parallel instead of -j 1.
# Result: ~/rb/NAME/av1gpu.  Prints RELBUILD_DONE.
T=$1; N=$2; IMG=${3:-nvidia/cuda:12.9.1-devel-ubuntu22.04}; XCM=$4   # XCM: extra CMake options
rm -rf ~/rb/$N && mkdir -p ~/rb/$N/src && tar -xzf $T -C ~/rb/$N/src
s=$(date +%s)
sudo docker run --rm -e XCM="$XCM" -v ~/rb/$N/src:/src -w /src $IMG bash -c '
  set -e
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip pkg-config g++ file \
    libjpeg-turbo8-dev libpng-dev libwebp-dev zlib1g-dev > /dev/null
  pip3 install -q cmake==3.31.6 ninja
  nvcc --version | tail -1
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAV1GPU_STATIC_DEPS=ON -DAV1GPU_TESTS=OFF $XCM | grep -E "CUDA|av1gpu"
  cmake --build build -j 16 | tail -2
  cp build/av1gpu av1gpu.rel && strip --strip-debug av1gpu.rel
  ldd av1gpu.rel | tee ldd.txt
  if grep -E "jpeg|png|webp|libz|cudart" ldd.txt; then echo "not linked in statically"; exit 1; fi
  ./av1gpu.rel --help > /dev/null && echo help ok
  chmod -R a+rwX /src' > ~/rb/$N/build.log 2>&1
echo "exit $? after $(( $(date +%s) - s )) s" >> ~/rb/$N/build.log
cp ~/rb/$N/src/av1gpu.rel ~/rb/$N/av1gpu 2>/dev/null
tail -12 ~/rb/$N/build.log; ls -la ~/rb/$N/av1gpu
echo RELBUILD_DONE
