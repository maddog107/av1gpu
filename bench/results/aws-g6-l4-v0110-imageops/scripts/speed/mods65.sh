#!/bin/bash
# mods65.sh: what the new module adds.  Fat-binary contents per package (cuobjdump), the module's size per architecture, and
# load / JIT time on this GPU.  Not a timing stage for the encoder; modload itself runs alone.
L=~/results/mods; mkdir -p $L; cd $L
CU=/usr/local/cuda-13.2/bin; export PATH=$CU:$PATH
list() { # name binary: the embedded images and their sizes; cubins go to a directory of their own, and the module that holds k_trim_measure is named
  local n=$1 b=$2; mkdir -p x_$n; ( cd x_$n; rm -f *; cuobjdump -xelf all $b > /dev/null 2>&1; cuobjdump -xptx all $b > /dev/null 2>&1; ls -l | awk 'NR>1{print $5, $NF}' > ../files_$n.txt
    grep -l k_trim_measure * 2>/dev/null > ../imageops_$n.txt
    # zero-compressed images need no decompression to grep; report what was found
  ); ls -l $b | awk '{print "'$n' binary bytes", $5}' >> sizes.txt
}
: > sizes.txt
list v0100 ~/rel/av1gpu-0.10.0-linux-x86_64/av1gpu
list prci ~/pkg/pr13/av1gpu-dev-e779036-linux-x86_64/av1gpu
list v0100_12 ~/rel/av1gpu-0.10.0-linux-x86_64-cuda12/av1gpu
list prci_12 ~/pkg/pr12/av1gpu-dev-e779036-linux-x86_64-cuda12/av1gpu
[ -e ~/rel13/main.bin ] && list mainrel ~/rel13/main.bin
[ -e ~/rel13/pr.bin ] && list prrel ~/rel13/pr.bin
list main89 ~/main/build/av1gpu
list pr89 ~/pr/build/av1gpu
python3 - <<'PY' > summary.txt
import collections, os, re
for n in ("v0100", "prci", "mainrel", "prrel", "v0100_12", "prci_12", "main89", "pr89"):
    f = f"files_{n}.txt"
    if not os.path.exists(f): continue
    by = collections.Counter(); cnt = collections.Counter()
    for l in open(f):
        sz, nm = l.split(); m = re.search(r"sm_(\d+)(\.ptx|\.cubin)?", nm); k = (m.group(1) if m else "?") + ("ptx" if nm.endswith(".ptx") else "")
        by[k] += int(sz); cnt[k] += 1
    im = open(f"imageops_{n}.txt").read().split() if os.path.exists(f"imageops_{n}.txt") else []
    print(n, {k: (cnt[k], round(by[k] / 1048576, 2)) for k in sorted(by)})
    print("   k_imageops images:", [(x, os.path.getsize(f"x_{n}/{x}")) for x in im])
PY
gcc ~/modload.c -I/usr/local/cuda-13.2/include -lcuda -o ~/modload 2>> $L/gcc.log
# the sm_89 module of the CUDA 13.2 build: load time, eager and lazy, alongside the small modules of main for scale
IM=$(head -1 imageops_pr89.txt)
for mode in EAGER LAZY; do CUDA_MODULE_LOADING=$mode ~/modload x_pr89/$IM x_pr89/$IM x_pr89/$IM > modload_$mode.txt 2>&1; done
# JIT: k_imageops.cu and two yardsticks as sm_89 PTX, no compute cache, each loaded 3 times
cd ~/pr/build
for m in k_imageops k_misc k_lossless; do
  c=$(ninja -t commands CMakeFiles/av1gpu_encoder.dir/encoder/src/$m.cu.o | tail -1 | sed 's/^.*nvcc/\/usr\/local\/cuda\/bin\/nvcc/; s/ -MD.*$//; s/ -x cu .*$//; s/ -c .*$//')
  echo "$c" >> $L/ptx_cmd.txt
  ( $c -ptx --gpu-architecture=sm_89 -x cu ~/pr/encoder/src/$m.cu -o $L/$m.sm_89.ptx ) >> $L/ptx.log 2>&1
done
cd $L; for m in k_imageops k_misc k_lossless; do ls -l $m.sm_89.ptx | awk '{print $5, $NF}'; CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=EAGER ~/modload $m.sm_89.ptx $m.sm_89.ptx $m.sm_89.ptx; done > jit.txt 2>&1
echo MODS_DONE
