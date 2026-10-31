#!/bin/bash
# Stage 3: record the toolchain, the CPU, and the exact flags each build used (cache variables and one compile
# and link command per project, from ninja's build graph).  Writes ~/out/flags.log and ~/out/flags.json.
mkdir -p ~/out; cd ~/src
{
echo "## toolchain"; gcc --version | head -1; g++ --version | head -1; ld --version | head -1; cmake --version | head -1
echo "ninja $(ninja --version)"; nasm -v; yasm --version | head -1; uname -r; grep -m1 PRETTY_NAME /etc/os-release
echo "## cpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L2|L3|NUMA node\(s\)"
echo "avx2: $(grep -c -m1 -w avx2 /proc/cpuinfo) avx512f: $(grep -c -m1 -w avx512f /proc/cpuinfo) fma: $(grep -c -m1 -w fma /proc/cpuinfo)"
echo "gcc -march=native resolves to: $(gcc -march=native -Q --help=target | grep -E '^\s+-march=' | awk '{print $2}')"
for b in aom-build aom-tuned svt-build svt-tuned svt-pgo avif-build avif-tuned avif-pgo; do
  echo; echo "## $b"
  grep -E "^(CMAKE_BUILD_TYPE|CMAKE_C_FLAGS|CMAKE_C_FLAGS_RELEASE|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|CMAKE_INTERPROCEDURAL_OPTIMIZATION|CMAKE_INSTALL_PREFIX|NATIVE|SVT_AV1_LTO|SVT_AV1_PGO|SVT_AV1_PGO_DIR|SVT_INTERNAL_PGO_GENERATE|SVT_INTERNAL_PGO_USE|AOM_EXTRA_C_FLAGS|AOM_EXTRA_CXX_FLAGS|ENABLE_AVX512|BUILD_SHARED_LIBS|CMAKE_OUTPUT_DIRECTORY|AOM_TARGET_CPU|CONFIG_RUNTIME_CPU_DETECT):" $b/CMakeCache.txt
  ninja -C $b -t commands > /tmp/cmds_$b.txt 2>/dev/null
  case $b in
    aom-*) pats="av1/encoder/encoder.c av1/encoder/x86/rdopt_avx2.c libaom.so" ;;
    svt-*) pats="Codec/enc_dec_process.c ASM_AVX2/ libSvtAv1Enc.so" ;;
    avif-*) pats="src/avif.c libavif.so apps/avifenc.c" ;;
  esac
  for p in $pats; do echo "-- $p:"; grep -m1 -F -- "$p" /tmp/cmds_$b.txt | grep -v -F "cmake -E" | cut -c1-1500; done
  echo "-- commands with -flto: $(grep -c -- '-flto' /tmp/cmds_$b.txt) of $(wc -l < /tmp/cmds_$b.txt); with -march=native: $(grep -c -- '-march=native' /tmp/cmds_$b.txt); with -mno-avx: $(grep -c -- '-mno-avx' /tmp/cmds_$b.txt); with -fprofile-use: $(grep -c -- '-fprofile-use' /tmp/cmds_$b.txt); with -O3: $(grep -c -- '-O3' /tmp/cmds_$b.txt)"
done
echo; echo "## installed binaries"
for d in opt opt-tuned opt-pgo; do echo "== ~/$d"; ~/$d/bin/avifenc --version | head -1; ldd ~/$d/bin/avifenc | grep -E "avif|aom|Svt|yuv|jpeg"; readelf -d ~/$d/bin/avifenc | grep -E "RPATH|RUNPATH"; done
ls -la ~/opt/lib/libaom.so* ~/opt-tuned/lib/libaom.so* ~/opt/lib/libSvtAv1Enc.so* ~/opt-tuned/lib/libSvtAv1Enc.so* ~/opt-pgo/lib/libSvtAv1Enc.so*
sha256sum ~/opt/lib/libaom.so.3.13.1 ~/opt-tuned/lib/libaom.so.3.13.1 ~/opt/lib/libSvtAv1Enc.so.4.2.0 ~/opt-tuned/lib/libSvtAv1Enc.so.4.2.0 ~/opt-pgo/lib/libSvtAv1Enc.so.4.2.0 2>&1
} > ~/out/flags.log 2>&1
echo FLAGS_DONE
