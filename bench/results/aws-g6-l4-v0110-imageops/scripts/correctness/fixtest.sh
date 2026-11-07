#!/bin/bash
# build the patched copy and repeat the failing cases against it
rm -rf ~/prfix; cp -r ~/pr ~/prfix; rm -rf ~/prfix/build
python3 ~/t/fix.py || exit 1
cmake -S ~/prfix -B ~/prfix/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DAV1GPU_TESTS=ON > ~/results/fix.cfg.log 2>&1
cmake --build ~/prfix/build -j 16 > ~/results/fix.build.log 2>&1 || { tail -20 ~/results/fix.build.log; exit 1; }
export T=/dev/shm/pf; source ~/t/try.sh; B=~/prfix/build/av1gpu
echo "== F1"; for o in 1 6 8; do try $B ~/adv/jpeg/ex${o}_420_160x96.jpg $T/o.avif --resize 50%; done; try $B ~/adv/trim/bar_mid_200x120.png $T/o.avif --resize x50%
echo "== F2/F3"; G=$(ls ~/bp/e/*.png | head -8)
for g in 0.5 1x65536! 70000x1!; do { echo "$G" | head -3; echo "$(echo "$G" | sed -n 4p)	resize=$g"; echo "$G" | tail -4; } > $T/l; rm -rf $T/o; timeout 120 $B --batch $T/l --outdir $T/o > $T/out.txt 2> $T/err.txt; echo "resize=$g: rc=$? files=$(ls $T/o 2>/dev/null | wc -l)/8 $(grep -m1 skipped $T/err.txt | cut -c1-140)"; done
try $B ~/adv/trim/bar_mid_200x120.png $T/o.avif --resize 70000x1!
echo "== F4"; for o in "--yuv 444 --negate" "--grayscale Average" "--lossless --negate" "--grayscale Average --lossless"; do try $B ~/adv/tall/t_16x65535.png $T/o.avif --preset turbo3 $o; done
echo "== F5"; try $B ~/adv/trim/vsplit_256x64.png $T/o.avif --trim --skip-blank 99; try $B ~/adv/trim/hsplit_256x64.png $T/o.avif --trim --skip-blank 99; try $B ~/adv/flat/white_64x64.png $T/o.avif --trim --skip-blank 99
echo "== F6"; try $B ~/images_std/06abed7823b7f37f.jpg $T/o.avif --resize @2000000 --max-mp 100
cd ~/prfix/build && PATH=$HOME/im/bin:$PATH LD_LIBRARY_PATH=$HOME/im/lib ctest 2>&1 | tail -3
BIN=~/prfix/build/av1gpu python3 ~/t/exifgeo.py 2>&1 | tail -4 | cut -c1-170
