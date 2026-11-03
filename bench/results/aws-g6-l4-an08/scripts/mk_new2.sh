#!/bin/bash
# mk_new2.sh: ~/new2 = ~/new's sources + the uploaded ~/up2/*.cu* files, built for sm_89 (tests OFF) -> ~/results/new2.log
mkdir -p ~/new2
rsync -a --exclude build --exclude build_li ~/new/ ~/new2/
cp ~/up2/* ~/new2/encoder/src/
echo "new2 (branch + change 2)" > ~/new2/COMMIT
s=$(date +%s)
[ -f ~/new2/build/build.ninja ] || cmake -S ~/new2 -B ~/new2/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF > ~/results/new2.cfg.log 2>&1
cmake --build ~/new2/build -j 24 > ~/results/new2.log 2>&1
echo "exit $? after $(( $(date +%s) - s )) s" >> ~/results/new2.log
tail -n 2 ~/results/new2.log
