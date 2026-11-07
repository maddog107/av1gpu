#!/bin/bash
# pr65-perf instance setup: tools, main (~/main) and PR (~/pr) built for sm_89 with identical flags, image sets.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/main ~/pr ~/rel ~/work
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config libjpeg-turbo8-dev libpng-dev libwebp-dev webp python3-numpy python3-pil \
  librsvg2-bin fonts-dejavu-core build-essential dav1d aom-tools ccache rsync imagemagick time bc \
  cuda-nsight-systems-13-2 > $L/apt.log 2>&1
for p in cuda-cuobjdump-13-2 cuda-nvdisasm-13-2 cuda-cuobjdump cuda-nvdisasm; do sudo apt-get install -yq $p >> $L/apt.log 2>&1; done
apt-cache search -n 'cuobjdump|nvdisasm' > $L/objdump_pkgs.txt
tar -xzf ~/main.tar.gz -C ~/main; tar -xzf ~/pr65.tar.gz -C ~/pr
CFG="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DAV1GPU_TESTS=OFF"
( s=$(date +%s); cmake -S ~/main -B ~/main/build $CFG > $L/main.cfg.log 2>&1
  cmake --build ~/main/build -j 16 > $L/main.log 2>&1; echo "main exit $? after $(( $(date +%s) - s )) s" >> $L/main.log
  s=$(date +%s); cmake -S ~/pr -B ~/pr/build $CFG -DAV1GPU_RECON_OBJECTS=$HOME/main/build/CMakeFiles/av1gpu_recon.dir/encoder/src > $L/pr.cfg.log 2>&1
  cmake --build ~/pr/build -j 16 > $L/pr.log 2>&1; echo "pr exit $? after $(( $(date +%s) - s )) s" >> $L/pr.log ) &
( cd ~/main && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  { head -1 bench/palette_study_images.csv; awk -F, 'NR>1 && /,comics,https/' bench/palette_study_images.csv; } > ~/comics.csv
  python3 bench/fetch_images.py --manifest ~/comics.csv --out ~/sets/comics --jobs 3 > $L/fetch_comics.log 2>&1
  python3 ~/mk_perf_sets.py > $L/sets.log 2>&1; echo "sets exit $?" >> $L/sets.log ) &
( cd ~/rel; G=https://github.com/maddog107/av1gpu/releases/download/v0.10.0
  for s in "" -cuda12; do f=av1gpu-0.10.0-linux-x86_64$s.tar.gz
    curl -fsSLO $G/$f && curl -fsSLO $G/$f.sha256 && sha256sum -c $f.sha256 && tar xzf $f; done > $L/rel.log 2>&1 ) &
wait
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -2 | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds; gcc --version | head -1; cmake --version | head -1
  which nsys cuobjdump nvdisasm; ls /opt/nvidia/ 2>/dev/null
  for d in ~/images ~/images_std ~/sets/*/; do echo "$d: $(ls $d | wc -l) files, $(du -sm $d | cut -f1) MB"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/main.log $L/pr.log $L/fetch.log $L/fetch_std.log $L/fetch_comics.log $L/sets.log $L/rel.log
echo SETUP_DONE
