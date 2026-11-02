#!/bin/bash
# probe4.sh: which sm_120 cubins the 13.4 UMD loads as they are (no compute-cache entry) and which it finalizes
# again; also the target the cache entry names
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-4 cuda-cudart-dev-13-4 cuda-cuobjdump-13-4 > /dev/null 2>&1
ls -d /usr/local/cuda-13.4/bin/nvcc
cd ~/p2
/usr/local/cuda-12.9/bin/nvcc --list-gpu-arch | tr '\n' ' '; echo
/usr/local/cuda-12.9/bin/nvcc --list-gpu-code | tr '\n' ' '; echo
b() { local tk=$1 name=$2; shift 2; /usr/local/cuda-$tk/bin/nvcc -O3 -cudart static "$@" -o v_$name k.cu 2>&1 | head -3; }
b 12.9 129_120       -gencode arch=compute_120,code=sm_120
b 12.9 129_120a      -gencode arch=compute_120a,code=sm_120a
b 12.9 129_120f      -gencode arch=compute_120f,code=sm_120f
b 12.9 129_120+120a  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120a,code=sm_120a
b 12.9 129_120+120f  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120f,code=sm_120f
b 12.9 129_120a+ptx  -gencode arch=compute_120a,code=sm_120a -gencode arch=compute_120,code=[sm_120,compute_120]
b 12.8 128_120       -gencode arch=compute_120,code=sm_120
b 13.4 134_120       -gencode arch=compute_120,code=sm_120
b 13.4 134_120a      -gencode arch=compute_120a,code=sm_120a
b 13.4 134_120f      -gencode arch=compute_120f,code=sm_120f
for v in v_*; do
  for u in 595 13.4; do
    [ $u = 595 ] && L= || L=/usr/local/cuda-$u/compat
    rm -rf c; r=$(env LD_LIBRARY_PATH=$L CUDA_CACHE_PATH=$PWD/c ./$v 2>&1)
    e=$(find c -type f ! -name index 2>/dev/null | head -1)
    tgt=$([ -n "$e" ] && head -c 200 $e | strings -n 4 | grep -oE 'sm_[0-9]+[af]?' | head -1)
    printf '%-22s UMD %-4s %-10s cache entries %s %s\n' $v $u "$r" $(find c -type f ! -name index 2>/dev/null | wc -l) "$tgt"
  done
done
echo "== strings in ptxas / libcuda about finalization"
strings -n 8 /usr/local/cuda-12.9/bin/ptxas | grep -iE "^-?-?[a-z-]*(merc|final)[a-z-]*$" | sort -u | head -20
strings -n 8 /usr/local/cuda-13.4/compat/libcuda.so.615.71.10 | grep -E "^CUDA_[A-Z_]+$" | sort -u > umd134_env.txt
strings -n 8 /usr/local/cuda-13.3/compat/libcuda.so.610.57.04 | grep -E "^CUDA_[A-Z_]+$" | sort -u > umd133_env.txt
echo "env vars new in 13.4:"; comm -13 umd133_env.txt umd134_env.txt
grep -iE "FINAL|MERC|SPECIAL|JIT|SASS" umd134_env.txt
