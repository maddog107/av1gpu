#!/bin/bash
# avifenc 1.4.0 with libaom 3.13.1 (as the published baselines) in ~/opt2, then for the README's two size-matched crop
# figures (car, dog): the highest avifenc -q at speeds 6 and 9 (tune=iq) whose file is no larger than av1gpu's new
# q60 file (as scratchpad tools/aom_match.sh did).  Output ~/fig/{src,av1gpu,aom}.  Prints AOM_DONE.
set -e
L=~/results/setup; mkdir -p ~/src ~/opt2 ~/fig/src ~/fig/av1gpu ~/fig/aom
cd ~/src
[ -d aom ] || git clone -q --depth 1 -b v3.13.1 https://aomedia.googlesource.com/aom
cmake -S aom -B aom-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt2 -DENABLE_TESTS=0 -DENABLE_DOCS=0 \
  -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=1 > $L/aom.log 2>&1
ninja -C aom-build install >> $L/aom.log 2>&1
PKG_CONFIG_PATH=$HOME/opt2/lib/pkgconfig cmake -S libavif -B avif2-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt2 \
  -DAVIF_CODEC_AOM=SYSTEM -DAVIF_CODEC_DAV1D=OFF -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM \
  -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_PREFIX_PATH=$HOME/opt2 -DCMAKE_INSTALL_RPATH=$HOME/opt2/lib > $L/avif2.log 2>&1
ninja -C avif2-build install >> $L/avif2.log 2>&1
E=~/opt2/bin/avifenc; $E --version | head -1
cd ~/fig
for n in b1aa6e8182e3f22f_huge df28dcbd0a84d456_huge; do
  cp ~/images/$n.jpg src/; cp ~/enc/new/gamut_default/$n.jpg.q120.avif av1gpu/$n.jpg.avif
  t=$(stat -c %s av1gpu/$n.jpg.avif)
  for s in 6 9; do
    lo=0; hi=100; best=0
    while [ $lo -le $hi ]; do
      q=$(( (lo + hi) / 2 ))
      $E -q $q -s $s -y 420 -j 8 -a tune=iq --ignore-exif --ignore-xmp src/$n.jpg aom/t.avif >/dev/null
      if [ $(stat -c %s aom/t.avif) -le $t ]; then best=$q; lo=$((q + 1)); else hi=$((q - 1)); fi
    done
    $E -q $best -s $s -y 420 -j 8 -a tune=iq --ignore-exif --ignore-xmp src/$n.jpg aom/$n.s$s.avif >/dev/null
    echo "$n s$s q$best $(stat -c %s aom/$n.s$s.avif) target $t"
  done
done
# control: the published crops' libaom files, from the v0.3.0 (2x2) av1gpu sizes
mkdir -p aom_old
for n in b1aa6e8182e3f22f_huge df28dcbd0a84d456_huge; do
  t=$(stat -c %s ~/enc/old/gamut_default/$n.jpg.q120.avif)
  for s in 6 9; do
    lo=0; hi=100; best=0
    while [ $lo -le $hi ]; do
      q=$(( (lo + hi) / 2 ))
      $E -q $q -s $s -y 420 -j 8 -a tune=iq --ignore-exif --ignore-xmp src/$n.jpg aom/t.avif >/dev/null
      if [ $(stat -c %s aom/t.avif) -le $t ]; then best=$q; lo=$((q + 1)); else hi=$((q - 1)); fi
    done
    $E -q $best -s $s -y 420 -j 8 -a tune=iq --ignore-exif --ignore-xmp src/$n.jpg aom_old/$n.s$s.avif >/dev/null
    echo "control $n s$s q$best $(stat -c %s aom_old/$n.s$s.avif) target $t $(md5sum < aom_old/$n.s$s.avif | cut -c1-12)"
  done
done
rm -f aom/t.avif
echo AOM_DONE
