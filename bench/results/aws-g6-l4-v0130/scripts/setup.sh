#!/bin/bash
# v0.13.0 docs refresh, instance setup (AWS g6.4xlarge, NVIDIA L4, Deep Learning Base AMI Ubuntu 24.04, CUDA 13.2):
#   packages; the release source tree (~/src.tar.gz -> ~/src, also ~/br for the set builders); the binaries
#   NEW = published v0.13.0 Linux CUDA 13 package, OLD = published v0.12.0 one (downloaded, sha256sum -c against the
#   release's .sha256; NEW_BIN=/path or OLD_BIN=/path stands in a local build; NEW_VER / OLD_VER pick other tags);
#   tools: libavif 1.4.0 avifdec (system dav1d), libjxl 0.11.1 ssimulacra2 (system lcms2, as the published scores),
#   ~/venv with cvvdp 0.5.7 + torch 2.15.0 (the versions of the published CVVDP run, aws-g6-l4-tiles4x4);
#   the size-range photos (~/images, gamut manifest) and the standard set (~/images_std) via bench/fetch_images.py;
#   the generated sets of the docs speed list (aws-g6-l4-speed-v06/scripts: mk_ll_sets.py -> ~/ll incl. Kodak,
#   mk_sets.py -> ~/sets incl. thumbnails, mk_mono.sh); the single-image start-up inputs ~/st (as the v0.9.0 check);
#   the RGBA set from ~/data.tar.gz if uploaded (else every alpha part is skipped);
#   sha256 of every input (~/results/inputs.sha256, comparable with aws-g6-g7-v090-check/logs/l4/inputs.sha256); env.
# Uploaded beforehand: ~/src.tar.gz (git archive of the release), ~/v13/ (this directory), optionally ~/data.tar.gz.
# Logs in ~/results/setup.  Prints SETUP_DONE, or SETUP_FAILED: <why> lines (exit 1).  Re-runnable.
set -x
export DEBIAN_FRONTEND=noninteractive
NEW_VER=${NEW_VER:-0.13.0}; OLD_VER=${OLD_VER:-0.12.0}
L=~/results/setup; mkdir -p $L ~/src ~/rel ~/bin ~/opt/bin ~/tools ~/st ~/images ~/images_std
T0=$(date +%s)
FAIL() { echo "SETUP_FAILED: $*"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || FAIL "nvidia-smi"
[ -f ~/src.tar.gz ] || FAIL "no ~/src.tar.gz"
sudo apt-get update -q >/dev/null
sudo apt-get install -yq cmake ninja-build pkg-config build-essential git curl libjpeg-turbo8-dev libpng-dev libwebp-dev webp \
  python3-numpy python3-pil python3-venv python3-pip librsvg2-bin fonts-dejavu-core dav1d libdav1d-dev aom-tools imagemagick \
  libyuv-dev libgif-dev libbrotli-dev libhwy-dev liblcms2-dev nasm bc xz-utils > $L/apt.log 2>&1 || FAIL "apt-get (see $L/apt.log)"
echo "apt done after $(( $(date +%s) - T0 )) s"
tar -xzf ~/src.tar.gz -C ~/src || FAIL "~/src.tar.gz"
ln -sfn ~/src ~/br
P=~/src/bench/results/aws-g6-l4-speed-v06/scripts
[ -f $P/mk_ll_sets.py ] && [ -f ~/src/bench/fetch_images.py ] && [ -f ~/src/tests/decoder_gate.py ] || FAIL "~/src is not an av1gpu tree"
if [ -f ~/data.tar.gz ]; then
  mkdir -p ~/data && tar -xzf ~/data.tar.gz -C ~/data && gcc -O2 -o ~/data/tools/pngraw ~/data/tools/pngraw.c -lpng > $L/pngraw.log 2>&1
  echo "RGBA set: $(ls ~/data/rgba/set 2>/dev/null | wc -l) files"
else
  echo "SETUP_WARNING: no ~/data.tar.gz: the RGBA set is absent, every alpha part will be skipped"
fi
# --- background jobs: photos, venv, ssimulacra2, avifdec, release packages
( cd ~/src && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1
  echo "fetch: $(ls ~/images/*.jpg | wc -l) $(ls ~/images_std/*.jpg | wc -l)" >> $L/fetch.log ) &
( [ -x ~/venv/bin/python ] || python3 -m venv ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install cvvdp==0.5.7 torch==2.15.0 numpy==2.5.4 pillow==13.0.0 scipy==1.18.1 > $L/pip.log 2>&1 ||
    ~/venv/bin/pip install cvvdp==0.5.7 >> $L/pip.log 2>&1
  echo "pip exit $?" >> $L/pip.log ) &
( [ -x ~/opt/bin/ssimulacra2 ] && exit 0
  # the published scores' build (tiles4x4/scripts/setup.sh): system highway / brotli / lcms2, no skcms.  Fallback, flagged
  # in env.txt and summary.md, only if that build fails: deps.sh + skcms (setup_p2.sh's recipe)
  cd ~/tools && rm -rf libjxl && git clone -q --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl && cd libjxl
  nice cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \
    -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \
    -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF > $L/jxl.cfg.log 2>&1
  nice cmake --build build --target ssimulacra2 -j 8 > $L/jxl.log 2>&1; echo "ssimulacra2 exit $?" >> $L/jxl.log
  echo lcms2 > ~/opt/ssimulacra2.cms
  if [ ! -x build/tools/ssimulacra2 ]; then
    ./deps.sh > $L/jxl.deps.log 2>&1; rm -rf build
    nice cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON -DJPEGXL_ENABLE_BENCHMARK=OFF \
      -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF \
      -DJPEGXL_ENABLE_SKCMS=ON >> $L/jxl.cfg.log 2>&1
    nice cmake --build build --target ssimulacra2 -j 8 >> $L/jxl.log 2>&1; echo "ssimulacra2 (skcms fallback) exit $?" >> $L/jxl.log
    echo "skcms (FALLBACK: not the published scores' build)" > ~/opt/ssimulacra2.cms
  fi
  cp build/tools/ssimulacra2 ~/opt/bin/ 2>> $L/jxl.log ) &
( [ -x ~/opt/bin/avifdec ] && exit 0
  set -e; cd ~/tools; rm -rf libavif avif-build
  git clone -q --depth 1 -b v1.4.0 https://github.com/AOMediaCodec/libavif
  cmake -S libavif -B avif-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt \
    -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_BUILD_APPS=ON -DAVIF_LIBYUV=SYSTEM -DAVIF_LIBSHARPYUV=OFF -DAVIF_JPEG=SYSTEM \
    -DAVIF_ZLIBPNG=SYSTEM -DCMAKE_INSTALL_RPATH=$HOME/opt/lib > $L/avif.log 2>&1
  ninja -C avif-build install >> $L/avif.log 2>&1; echo "avif exit 0" >> $L/avif.log ) &
getrel() {   # version wait_minutes: the Linux CUDA 13 package, sha256-checked, unpacked in ~/rel
  local v=$1 w=$2 G=https://github.com/maddog107/av1gpu/releases/download/v$1 f=av1gpu-$1-linux-x86_64.tar.gz i
  [ -x ~/rel/av1gpu-$v-linux-x86_64/av1gpu ] && return 0
  for i in $(seq 0 $w); do curl -fsSI $G/$f.sha256 > /dev/null && break; [ $i = $w ] && return 1; sleep 60; done
  ( cd ~/rel && curl -fsSLO $G/$f && curl -fsSLO $G/$f.sha256 && sha256sum -c $f.sha256 && tar -xzf $f )
}
( if [ -z "$OLD_BIN" ]; then getrel $OLD_VER 0; echo "old exit $?"; fi
  if [ -z "$NEW_BIN" ]; then getrel $NEW_VER ${REL_WAIT_MIN:-60}; echo "new exit $?"; fi ) > $L/rel.log 2>&1 &
wait
echo "background jobs done after $(( $(date +%s) - T0 )) s"
NB=${NEW_BIN:-$HOME/rel/av1gpu-$NEW_VER-linux-x86_64/av1gpu}; OB=${OLD_BIN:-$HOME/rel/av1gpu-$OLD_VER-linux-x86_64/av1gpu}
ln -sfn $NB ~/bin/new; ln -sfn $OB ~/bin/old
[ -x $OB ] && cp $OB ~/bin/old2   # the noise binary of the start-up timing: v0.12.0 again, from a second path
{ echo "new $NB $(sha256sum < $NB | cut -c1-64) $($NB --version 2>&1 | head -1)"
  echo "old $OB $(sha256sum < $OB | cut -c1-64) $($OB --version 2>&1 | head -1)"
  sha256sum ~/rel/*.tar.gz 2>/dev/null; echo "NEW_BIN=${NEW_BIN:-} OLD_BIN=${OLD_BIN:-}"; } > ~/results/binaries.txt 2>&1
# --- generated sets (need ~/images, ~/br): lossless sets incl. Kodak, thumbnails / odd / gray / webp, mono, ~/st
( cd ~ && python3 $P/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 $P/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
( bash $P/mk_mono.sh > $L/mono.log 2>&1; echo "exit $?" >> $L/mono.log )
python3 - > $L/st.log 2>&1 <<'PY'
import glob, os, shutil
from PIL import Image
H = os.path.expanduser("~")
fs = sorted(glob.glob(f"{H}/images/*.jpg"))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
photo = min((f for f in fs if 2 <= mp[f] <= 3), key=lambda f: abs(mp[f] - 2.5))
big = min((f for f in fs if 10 <= mp[f] <= 12), key=lambda f: abs(mp[f] - 11))
shutil.copy(photo, f"{H}/st/photo.jpg"); shutil.copy(big, f"{H}/st/big.jpg")
for f in (photo, big):
    print(os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
PY
convert ~/st/photo.jpg -resize 512x384^ -gravity center -extent 512x384 -quality 85 ~/st/thumb512.jpg
t=$(ls ~/sets/thumbs/t320_*.jpg | head -1); cp $t ~/st/t320.jpg; echo "t320 from $(basename $t)" >> $L/st.log
printf "%s\n" ~/st/photo.jpg ~/st/thumb512.jpg > ~/st/pair.txt
( cd ~ && find images images_std ll sets st data/rgba/set -type f -o -type l 2>/dev/null | sort | while read f; do
    printf "%s  %s\n" "$(sha256sum < "$f" | cut -c1-64)" "$f"; done ) > ~/results/inputs.sha256
set +x
{ date -u; nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1
  ~/opt/bin/avifdec --version 2>&1 | head -3; echo "dav1d $(dav1d --version)"; aomdec --help 2>&1 | grep -m1 -i 'aom'
  echo "ssimulacra2: libjxl v0.11.1, colour management $(cat ~/opt/ssimulacra2.cms 2>/dev/null) $(ls -la ~/opt/bin/ssimulacra2 2>&1)"
  ~/venv/bin/pip list 2>/dev/null | grep -i -E "^(cvvdp|torch|numpy|pillow|scipy) "
  ~/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python3 -c "import PIL, numpy; print('system pillow', PIL.__version__, 'numpy', numpy.__version__)"
  convert -version | head -1
  echo "nproc $(nproc)"; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds
  cat ~/results/binaries.txt
  echo "rgba: $(ls ~/data/rgba/set 2>/dev/null | wc -l) files"
  for d in ~/images ~/images_std ~/ll/*/ ~/sets/*/ ~/st; do echo "$d: $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/*.log
OK=1
bad() { echo "SETUP_FAILED: $*"; OK=""; }
[ -x ~/bin/new ] || bad "no NEW binary (v$NEW_VER package not published yet? see $L/rel.log; or set NEW_BIN)"
[ -x ~/bin/old ] || bad "no OLD binary (see $L/rel.log; or set OLD_BIN)"
~/bin/new --version > /dev/null 2>&1 || bad "NEW binary does not run"
[ "$(ls ~/images/*.jpg 2>/dev/null | wc -l)" = 100 ] || bad "~/images does not hold 100 photos ($L/fetch.log)"
[ "$(ls ~/images_std/*.jpg 2>/dev/null | wc -l)" = 100 ] || bad "~/images_std does not hold 100 photos ($L/fetch_std.log)"
[ -x ~/opt/bin/ssimulacra2 ] || bad "no ssimulacra2 ($L/jxl.log)"
[ -x ~/opt/bin/avifdec ] || bad "no avifdec ($L/avif.log)"
~/venv/bin/python -c "import pycvvdp, torch; assert torch.cuda.is_available()" 2>/dev/null || bad "venv: no pycvvdp / CUDA torch ($L/pip.log)"
[ "$(ls ~/ll/kodak/*.png 2>/dev/null | wc -l)" = 24 ] || bad "~/ll/kodak does not hold 24 PNGs ($L/llsets.log)"
[ -n "$(ls ~/ll/cmp/*.png 2>/dev/null)" ] || bad "no ~/ll/cmp ($L/llsets.log)"
[ -n "$(ls ~/sets/thumbs/*.jpg 2>/dev/null)" ] || bad "no ~/sets/thumbs ($L/sets.log)"
for f in photo.jpg big.jpg thumb512.jpg t320.jpg pair.txt; do [ -s ~/st/$f ] || bad "no ~/st/$f"; done
command -v dav1d > /dev/null && command -v aomdec > /dev/null || bad "dav1d / aomdec not on PATH (decoder gate)"
echo "setup took $(( $(date +%s) - T0 )) s"
[ -n "$OK" ] || exit 1
echo SETUP_DONE
