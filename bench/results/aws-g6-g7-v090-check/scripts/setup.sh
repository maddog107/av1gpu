#!/bin/bash
# rc09 instance setup (L4 and Blackwell alike): packages; main c553ffb's tree (~/src, also ~/br for mk_ll_sets.py);
# the v0.8.0 Linux release tarballs (CUDA 13 = A13, CUDA 12 = A12; sha256 checked against the .sha256 files);
# the identity sets of speed-v06 (size range ~/images, standard ~/images_std, ~/ll, ~/sets, mono) and the palette
# sets (~/pal/sets); the single-image inputs ~/st (a 2-3 MP and a 10-12 MP photo of the size range, its 512 x 384
# thumbnail, a 320 x 320 thumbnail); lists ~/lists; sha256 of every input (~/results/inputs.sha256); env.
# Logs in ~/results/setup.  Prints SETUP_DONE.
set -x
export DEBIAN_FRONTEND=noninteractive
L=~/results/setup; mkdir -p $L ~/src ~/data ~/pal ~/rel ~/bin ~/st ~/lists ~/render
sudo apt-get update -q >/dev/null
sudo apt-get install -yq python3-numpy python3-pil librsvg2-bin fonts-dejavu-core dav1d aom-tools imagemagick webp bc \
  > $L/apt.log 2>&1
tar -xzf ~/src.tar.gz -C ~/src; tar -xzf ~/data.tar.gz -C ~/data; tar -xzf ~/render.tgz -C ~/render; ln -sfn ~/src ~/br
G=https://github.com/maddog107/av1gpu/releases/download/v0.8.0
( cd ~/rel && for s in "" -cuda12; do f=av1gpu-0.8.0-linux-x86_64$s.tar.gz
    curl -fsSLO $G/$f && curl -fsSLO $G/$f.sha256 && sha256sum -c $f.sha256 && tar -xzf $f; done
  ls -la ~/rel ) > $L/rel.log 2>&1
cp ~/rel/av1gpu-0.8.0-linux-x86_64/av1gpu ~/bin/A13; cp ~/rel/av1gpu-0.8.0-linux-x86_64-cuda12/av1gpu ~/bin/A12
cp ~/bin/A13 ~/bin/A13c   # the noise binary: v0.8.0 again, from a second path
( cd ~/src && python3 bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out ~/images > $L/fetch.log 2>&1
  python3 bench/fetch_images.py --out ~/images_std > $L/fetch_std.log 2>&1 )
P=~/src/bench/results/aws-g6-l4-speed-v06/scripts
( cd ~ && python3 $P/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log )
( cd ~ && python3 $P/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log )
( bash $P/mk_mono.sh > $L/mono.log 2>&1; echo "exit $?" >> $L/mono.log )
( cd ~ && python3 ~/src/bench/results/aws-g6-l4-palette-study/scripts/prep_sets.py ~/src/bench/palette_study_images.csv \
    ~/src/bench/openimages_v4_gamut.csv ~/src ~/render ~/pal > $L/pal.log 2>&1; echo "exit $?" >> $L/pal.log )
# single-image inputs
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
identify ~/st/*.jpg >> $L/st.log
printf "%s\n" ~/st/photo.jpg ~/st/thumb512.jpg > ~/st/pair.txt
# lists
mkdir -p ~/lists/short; split -l 8 -d ~/sets/mixed.txt ~/lists/short/mixed_
cp ~/st/pair.txt ~/lists/short/pair.txt
{ ls ~/sets/thumbs/* | awk 'NR % 28 == 1'; ls ~/images/*.jpg | awk 'NR % 10 == 5'; ls ~/images_std/*.jpg | awk 'NR % 10 == 2'
  ls ~/data/rgba/set/* | head -5; ls ~/sets/gray/*; ls ~/sets/odd/* | awk 'NR % 3 == 1'; ls ~/sets/webp/*
  ls ~/pal/sets/screen/* | head -6; ls ~/pal/sets/flat_png/* | head -4; ls ~/pal/sets/flat_jpg/* | head -4
  ls ~/sets/mono/* | awk 'NR % 6 == 1'; ls ~/ll/png16_10/* | head -3; ls ~/ll/misc/* | head -6; ls ~/ll/jpeg/* | awk 'NR % 3 == 1'
  ls ~/ll/kodak/* | head -2; } > ~/lists/single_all.txt
ls ~/sets/single/* > ~/lists/single10.txt
# ~30 files for the Blackwell single-file run: every 4th of single_all
awk 'NR % 4 == 1' ~/lists/single_all.txt > ~/lists/single30.txt
wc -l ~/lists/*.txt ~/lists/short/* > $L/lists.log
( cd ~ && find images images_std ll sets pal/sets data/rgba/set st -type f -o -type l | sort | while read f; do
    printf "%s  %s\n" "$(sha256sum < "$f" | cut -c1-64)" "$f"; done ) > ~/results/inputs.sha256
{ nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,compute_cap --format=csv,noheader
  /usr/local/cuda/bin/nvcc --version | tail -1; /usr/local/cuda/bin/compute-sanitizer --version | tail -1
  python3 -c "import PIL, numpy; print('pillow', PIL.__version__, 'numpy', numpy.__version__)"
  echo "dav1d $(dav1d --version)"; aomdec --help 2>&1 | grep -m1 -i 'aom'; convert -version | head -1
  nproc; lscpu | grep "Model name"; free -g | head -2; uname -r; lsb_release -ds
  sha256sum ~/rel/*.tar.gz ~/bin/A13 ~/bin/A12
  for d in ~/images ~/images_std ~/ll/*/ ~/sets/*/ ~/pal/sets/*/; do echo "$d: $(ls $d | wc -l) files"; done; } > ~/results/env.txt 2>&1
tail -n 2 $L/*.log
echo SETUP_DONE
