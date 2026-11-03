#!/bin/bash
# After setup.sh: (1) libaom 3.13.1 patched with the LRSTUDY switch (patch_lrstudy.py) + libavif 1.4.0 against it
# -> ~/opt-lr/bin/avifenc; (2) the inspection decoder lrstat (lrstat.c in place of examples/inspect.c) -> ~/opt/bin/lrstat.
# Prints BUILD_LR_DONE.
set -e
L=~/results/setup; cd ~/src
rm -rf aom-lr aom-lr-build avif-lr-build aom-insp aom-insp-build
cp -r aom aom-lr; python3 ~/bench/patch_lrstudy.py ~/src/aom-lr
cp -r aom aom-insp; cp ~/bench/lrstat.c aom-insp/examples/inspect.c
( cmake -S aom-insp -B aom-insp-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONFIG_AV1_ENCODER=0 -DCONFIG_INSPECTION=1 \
    -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=0 -DBUILD_SHARED_LIBS=0 > $L/aom-insp.log 2>&1
  ninja -C aom-insp-build -j 16 inspect >> $L/aom-insp.log 2>&1
  cp aom-insp-build/examples/inspect ~/opt/bin/lrstat 2>/dev/null || cp aom-insp-build/inspect ~/opt/bin/lrstat ) &
cmake -S aom-lr -B aom-lr-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt-lr -DENABLE_TESTS=0 -DENABLE_DOCS=0 \
  -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 > $L/aom-lr.log 2>&1
ninja -C aom-lr-build -j 48 install >> $L/aom-lr.log 2>&1
PKG_CONFIG_PATH=$HOME/opt-lr/lib/pkgconfig cmake -S libavif -B avif-lr-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt-lr -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM \
  -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_PREFIX_PATH=$HOME/opt-lr \
  -DCMAKE_INSTALL_RPATH=$HOME/opt-lr/lib > $L/avif-lr.log 2>&1
ninja -C avif-lr-build install >> $L/avif-lr.log 2>&1
wait
ldd ~/opt-lr/bin/avifenc | grep aom; ldd ~/opt/bin/avifenc | grep aom
ls -la ~/opt/bin/lrstat ~/opt-lr/bin/avifenc
echo BUILD_LR_DONE
