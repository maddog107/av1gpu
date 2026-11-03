#!/bin/bash
# modload.sh ARCH: extract every ARCH cubin (sm_89 / sm_120) of the v0.8.0 and v0.9.0 CUDA 13 binaries and time their
# loading on this GPU (modload.c), lazy (the default) and eager, 3 times each.  Needs cuobjdump 13.4 (apt
# cuda-cuobjdump-13-4).  ~/results/modload_ARCH.txt
A=$1; CU=/usr/local/cuda-13.4/bin; O=~/results/modload_$A.txt; : > $O
/usr/local/cuda-13.2/bin/nvcc -O2 -o ~/modload ~/modload.c -lcuda >> $O 2>&1
for b in A13 B13; do
  D=~/work/cubins_${b}_$A
  if [ ! -d $D ]; then mkdir -p $D; cd $D
    for n in $($CU/cuobjdump -lelf ~/bin/$b | grep "\.$A\.cubin" | awk '{print $NF}'); do $CU/cuobjdump -xelf $n ~/bin/$b > /dev/null; done
  fi
  cd $D
  echo "== $b ($A): $(ls | wc -l) cubins, $(du -sh . | cut -f1)" >> $O
  for k in 1 2 3; do for mode in LAZY EAGER; do CUDA_MODULE_LOADING=$mode ~/modload $(ls -S) >> $O 2>&1; done; done
done
echo MODLOAD_DONE >> $O
