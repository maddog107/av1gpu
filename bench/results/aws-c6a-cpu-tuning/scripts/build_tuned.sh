#!/bin/bash
# Stage 2: tuned builds of the same sources.
#   ~/opt-tuned : libaom + SVT-AV1 + libavif with -O3 (CMake Release) -march=native and LTO
#   ~/opt-pgo   : SVT-AV1 as in opt-tuned plus PGO (SVT's own SVT_AV1_PGO mechanism), trained with avifenc
#                 on bench/images (the 100-image ~1 MP set; timing uses the gamut set); libavif built
#                 against it and opt-tuned's libaom
set -ex
cd ~/src
P=$HOME/opt-tuned; Q=$HOME/opt-pgo
gcc --version | head -1
# libaom has no -march=native or LTO option of its own: its AOM_EXTRA_C(XX)_FLAGS hooks + CMake IPO (-flto)
cmake -S aom -B aom-tuned -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 \
  -DAOM_EXTRA_C_FLAGS=-march=native -DAOM_EXTRA_CXX_FLAGS=-march=native -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON >/dev/null
ninja -C aom-tuned install >/dev/null
# SVT-AV1: its own NATIVE (-march=native) and SVT_AV1_LTO options (separate output dir: SVT writes binaries
# to <source>/Bin/<type> by default, which the stock build uses)
cmake -S svt -B svt-tuned -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DNATIVE=ON -DSVT_AV1_LTO=ON -DCMAKE_OUTPUT_DIRECTORY=$HOME/src/svt-tuned/Bin/ >/dev/null
ninja -C svt-tuned install >/dev/null
# libavif has no such options: CMAKE_C(XX)_FLAGS + CMake IPO
avif() {  # avif BUILD_DIR PREFIX PKG_CONFIG_PATH PREFIX_PATH RPATH
  PKG_CONFIG_PATH=$3 cmake -S libavif -B $1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$2 \
    -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_SVT=SYSTEM -DAVIF_CODEC_DAV1D=OFF -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM \
    -DCMAKE_PREFIX_PATH="$4" -DCMAKE_INSTALL_RPATH="$5" \
    -DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON >/dev/null
  ninja -C $1 install >/dev/null
}
avif avif-tuned $P $P/lib/pkgconfig $P $P/lib
$P/bin/avifenc --version | head -1
# SVT-AV1 PGO: configure with SVT_AV1_PGO=ON, then run what its PGOCompileGen / PGOCompileUse targets run
# (reconfigure with SVT_INTERNAL_PGO_GENERATE / _USE, clean rebuild of SvtAv1EncApp and the library), with an
# avifenc training run in place of PGOGenerateProfile (which encodes y4m video with SvtAv1EncApp at preset 2)
PD=$HOME/src/svt-pgo/profile
cmake -S svt -B svt-pgo -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$Q -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DNATIVE=ON -DSVT_AV1_LTO=ON -DSVT_AV1_PGO=ON -DSVT_AV1_PGO_DIR=$PD -DCMAKE_OUTPUT_DIRECTORY=$HOME/src/svt-pgo/Bin/ >/dev/null
cmake svt-pgo -DSVT_INTERNAL_PGO_GENERATE=ON -DSVT_INTERNAL_PGO_USE=OFF >/dev/null
cmake --build svt-pgo --clean-first --target SvtAv1EncApp >/dev/null
LD_LIBRARY_PATH=$HOME/src/svt-pgo/Bin ldd $P/bin/avifenc | grep -i svt
LD_LIBRARY_PATH=$HOME/src/svt-pgo/Bin python3 ~/train_pgo.py $P/bin/avifenc ~/av1gpu/bench/images
echo "profile files: $(find $PD -name '*.gcda' | wc -l)"
test "$(find $PD -name '*.gcda' | wc -l)" -gt 0
cmake svt-pgo -DSVT_INTERNAL_PGO_GENERATE=OFF -DSVT_INTERNAL_PGO_USE=ON >/dev/null
cmake --build svt-pgo --clean-first --target SvtAv1EncApp > ~/svt-pgo-use.log 2>&1
echo "PGO use build: $(grep -c -i 'profile' ~/svt-pgo-use.log) lines mentioning profile; first few:"; grep -i -m5 'profile' ~/svt-pgo-use.log || true
ninja -C svt-pgo install >/dev/null
avif avif-pgo $Q "$Q/lib/pkgconfig:$P/lib/pkgconfig" "$Q;$P" "$Q/lib;$P/lib"
for d in opt opt-tuned opt-pgo; do echo "== $d"; ~/$d/bin/avifenc --version | head -1; ldd ~/$d/bin/avifenc | grep -E "avif|aom|Svt"; done
echo BUILD_DONE
