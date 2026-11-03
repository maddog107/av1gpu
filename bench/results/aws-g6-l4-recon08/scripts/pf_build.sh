# pf_build.sh: ~/pf = main + extra recon profile ticks (k_recon.cuh, batch.cpp from ~/r/pf/), av1gpu only; then profile
set -e
rm -rf ~/pf; mkdir -p ~/pf; tar -xzf ~/main.tar.gz -C ~/pf
cp ~/r/pf/k_recon.cuh ~/pf/encoder/src/; cp ~/r/pf/batch.cpp ~/pf/encoder/src/
s=$(date +%s)
cmake -S ~/pf -B ~/pf/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF -DAV1GPU_RECON_JOBS=4 \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache > ~/r/pf_cfg.log 2>&1
cmake --build ~/pf/build -j 24 --target av1gpu > ~/r/pf_build.log 2>&1
echo "built after $(( $(date +%s) - s )) s"
O=~/results/prof2; mkdir -p $O ~/work/p2
for set in images images_std; do
  for enc in "" 1; do
    GP_ENCODERS=$enc ~/pf/build/av1gpu --batch ~/$set --outdir ~/work/p2/$set --prof > $O/$set${enc:+_enc$enc}.log 2>&1
  done
done
grep -h "stage ms\|recon phases\|CTA-cycles" $O/*.log
echo PF_DONE
