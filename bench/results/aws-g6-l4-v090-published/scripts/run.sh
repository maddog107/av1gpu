#!/bin/bash
# v0.9.0 published-package check on an L4: the release's Linux tarballs (CUDA 13.4 and 12.9) against v0.8.0's.
# Why: the tag build's k_an_c.cu cubins differ from the tested c553ffb build for a few architectures (ptxas
# output not reproducible; PTX identical), sm_89 among them in the CUDA 12 package. So run the published files.
O=~/out; mkdir -p $O ~/bin; exec > $O/run.log 2>&1
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q >/dev/null; sudo apt-get install -yq dav1d aom-tools python3-pil python3-numpy librsvg2-bin fonts-dejavu-core bc >/dev/null 2>&1
mkdir -p ~/rel ~/src; tar xzf ~/src.tar.gz -C ~/src; tar xzf ~/render.tgz -C ~
cd ~/rel; G=https://github.com/maddog107/av1gpu/releases/download
for f in v0.8.0/av1gpu-0.8.0-linux-x86_64.tar.gz v0.8.0/av1gpu-0.8.0-linux-x86_64-cuda12.tar.gz \
         v0.9.0/av1gpu-0.9.0-linux-x86_64.tar.gz v0.9.0/av1gpu-0.9.0-linux-x86_64-cuda12.tar.gz; do curl -fsSLO $G/$f; done
sha256sum -c ~/expect.sha256 || { echo SHA_FAIL; exit 1; }
for f in *.tar.gz; do tar xzf $f; done
A13=~/rel/av1gpu-0.8.0-linux-x86_64/av1gpu; A12=~/rel/av1gpu-0.8.0-linux-x86_64-cuda12/av1gpu
B13=~/rel/av1gpu-0.9.0-linux-x86_64/av1gpu; B12=~/rel/av1gpu-0.9.0-linux-x86_64-cuda12/av1gpu
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; dav1d --version; uname -r
$B13 --version 2>&1 | head -2; $B12 --version 2>&1 | head -2
cd ~
python3 ~/src/bench/results/aws-g6-l4-palette-study/scripts/prep_sets.py ~/src/bench/palette_study_images.csv \
  ~/src/bench/openimages_v4_gamut.csv ~/src ~/render ~/pal | tail -8
python3 ~/src/bench/fetch_images.py --out ~/pal/std100 | tail -1
python3 ~/src/bench/fetch_images.py --manifest ~/src/bench/openimages_v4_gamut.csv --out ~/pal/sizerange | tail -1
P=~/pal/sets; mkdir -p ~/pal/thumbs
python3 - ~/pal/sizerange ~/pal/thumbs <<'PY'
# thumbnails as production makes them: 512 x 384 crops-and-downscales and fits within 320 x 320, JPEG q85
import os, sys
from PIL import Image, ImageOps
src, dst = sys.argv[1:3]
for f in sorted(os.listdir(src)):
    if not f.endswith('.jpg'): continue
    im = ImageOps.exif_transpose(Image.open(os.path.join(src, f))).convert('RGB')
    ImageOps.fit(im, (512, 384), Image.LANCZOS).save(os.path.join(dst, 'd512_' + f), quality=85)
    t = im.copy(); t.thumbnail((320, 320), Image.LANCZOS); t.save(os.path.join(dst, 't320_' + f), quality=85)
PY
L=$O/lists; mkdir -p $L
ls ~/pal/std100/*.jpg > $L/std100.txt; ls ~/pal/sizerange/*.jpg > $L/sizes.txt; ls ~/pal/thumbs/*.jpg > $L/thumbs.txt
ls $P/photo/*.png > $L/kodak.txt
for s in comics flat_png flat_jpg painted screen photo; do ls $P/$s/* > $L/$s.txt; done
cat $L/screen.txt $L/flat_png.txt > $L/scrflat.txt
cat $L/comics.txt $L/flat_png.txt $L/painted.txt $L/screen.txt $L/photo.txt $L/std100.txt $L/sizes.txt > $L/all.txt
head -12 $L/sizes.txt > $L/single.txt; head -6 $L/thumbs.txt >> $L/single.txt; grep -m3 d512_ $L/thumbs.txt | tail -1 >> $L/single.txt
head -3 $L/screen.txt >> $L/single.txt; head -2 $L/flat_jpg.txt >> $L/single.txt
wc -l $L/*.txt
echo PREP_DONE
TOT=0; IDN=0
pair() {   # name binA binB list args...: encode the list with both binaries (batch mode), cmp every output
  local n=$1 a=$2 b=$3 l=$4; shift 4
  for t in A B; do
    local bin=$a; [ $t = B ] && bin=$b; mkdir -p $O/$n/$t
    $bin --batch $l --outdir $O/$n/$t "$@" > $O/$n/$t.log 2>&1; echo "$?" > $O/$n/$t.rc
  done
  local tot=0 d=0
  for f in $O/$n/A/*.avif; do tot=$((tot + 1)); cmp -s $f $O/$n/B/$(basename $f) || { d=$((d + 1)); echo "  DIFF $n $(basename $f)"; }; done
  local na=$(ls $O/$n/A | wc -l) nb=$(ls $O/$n/B | wc -l)
  [ $na = $nb ] || echo "  COUNT MISMATCH $n $na $nb"
  TOT=$((TOT + tot)); IDN=$((IDN + tot - d))
  echo "PAIR $n: $((tot - d))/$tot identical (exit A $(cat $O/$n/A.rc) B $(cat $O/$n/B.rc); list $(wc -l < $l))"
}
single() {   # name binA binB args...: one process per file of single.txt, cmp every output and exit status
  local n=$1 a=$2 b=$3; shift 3; local tot=0 d=0
  mkdir -p $O/$n/A $O/$n/B
  for f in $(cat $L/single.txt); do
    local o=$(basename ${f%.*}).avif
    $a "$@" $f $O/$n/A/$o > /dev/null 2>&1; local ra=$?; $b "$@" $f $O/$n/B/$o > /dev/null 2>&1; local rb=$?
    tot=$((tot + 1)); if [ $ra = $rb ] && cmp -s $O/$n/A/$o $O/$n/B/$o; then :; else d=$((d + 1)); echo "  DIFF $n $o (exit $ra $rb)"; fi
  done
  TOT=$((TOT + tot)); IDN=$((IDN + tot - d))
  echo "SINGLE $n: $((tot - d))/$tot identical"
}
runset() {   # tag binA binB: the identity list
  local t=$1 a=$2 b=$3
  pair ${t}_std100 $a $b $L/std100.txt
  pair ${t}_std100_nosmall $a $b $L/std100.txt --small-preset none
  pair ${t}_sizes $a $b $L/sizes.txt
  pair ${t}_sizes_444 $a $b $L/sizes.txt --yuv 444
  pair ${t}_sizes_10 $a $b $L/sizes.txt --depth 10
  pair ${t}_sizes_444_10 $a $b $L/sizes.txt --yuv 444 --depth 10
  pair ${t}_sizes_slow $a $b $L/sizes.txt --preset slow
  pair ${t}_sizes_q90 $a $b $L/sizes.txt --quality 90
  pair ${t}_sizes_q30 $a $b $L/sizes.txt --quality 30
  pair ${t}_sizes_turbo3ns $a $b $L/sizes.txt --preset turbo3 --small-preset none
  pair ${t}_thumbs $a $b $L/thumbs.txt
  pair ${t}_thumbs_444_10 $a $b $L/thumbs.txt --yuv 444 --depth 10
  pair ${t}_kodak_ll $a $b $L/kodak.txt --lossless
  for s in photo comics painted flat_jpg; do pair ${t}_$s $a $b $L/$s.txt; done
  pair ${t}_scrflat $a $b $L/scrflat.txt
  pair ${t}_scrflat_on $a $b $L/scrflat.txt --palette on
  pair ${t}_scrflat_444_10 $a $b $L/scrflat.txt --yuv 444 --depth 10
  pair ${t}_scrflat_ll $a $b $L/scrflat.txt --lossless
  single ${t}_single $a $b
  single ${t}_single_444_10 $a $b --yuv 444 --depth 10
}
echo "== 1. v0.8.0 against v0.9.0, CUDA 13.4 packages"
runset c13 $A13 $B13; echo "SUBTOTAL c13: $IDN/$TOT"; T1=$TOT; I1=$IDN
echo "== 2. v0.8.0 against v0.9.0, CUDA 12.9 packages"
runset c12 $A12 $B12; echo "SUBTOTAL c12: $((IDN - I1))/$((TOT - T1))"; T2=$TOT; I2=$IDN
echo "== 3. v0.9.0: CUDA 13.4 against CUDA 12.9"
pair x_all $B13 $B12 $L/all.txt
pair x_thumbs $B13 $B12 $L/thumbs.txt
pair x_sizes_444_10 $B13 $B12 $L/sizes.txt --yuv 444 --depth 10
pair x_scrflat_ll $B13 $B12 $L/scrflat.txt --lossless
single x_single $B13 $B12
echo "SUBTOTAL x: $((IDN - I2))/$((TOT - T2))"
echo "TOTAL: $IDN/$TOT identical"
echo "== 4. decoder gate (dav1d + aomdec), both v0.9.0 packages"
GO=$O/gate; mkdir -p $GO/photo $GO/thumb $GO/scr
for f in $(head -30 $L/sizes.txt); do ln -sf $f $GO/photo/; done
for f in $(head -40 $L/thumbs.txt); do ln -sf $f $GO/thumb/; done
for f in $(cat $L/scrflat.txt); do ln -sf $f $GO/scr/; done
g() { local n=$1 bin=$2 d=$3; shift 3; python3 ~/src/tests/decoder_gate.py --bin $bin --images $d --jobs 8 "$@" > $GO/$n.log 2>&1
  echo "GATE $n exit $?: $(grep -a -E '[0-9]+/[0-9]+' $GO/$n.log | tail -1)"; }
for v in 13 12; do
  bin=$B13; [ $v = 12 ] && bin=$B12
  g c${v}_photo $bin $GO/photo --q 60,120,200
  g c${v}_photo_444_10 $bin $GO/photo --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10
  g c${v}_thumb $bin $GO/thumb --q 60,120
  g c${v}_scr $bin $GO/scr --q 120,40
  g c${v}_scr_ll $bin $GO/scr --lossless
done
echo "== 5. compute-sanitizer (memcheck, racecheck), both v0.9.0 packages"
CS=$(ls /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | tail -1); echo "sanitizer $CS ($($CS --version | tail -1))"
head -3 $L/sizes.txt > $L/san3.txt
for v in 13 12; do
  bin=$B13; [ $v = 12 ] && bin=$B12
  for tool in memcheck racecheck; do
    for j in batch photo thumb; do
      case $j in
        batch) $CS --tool $tool --print-limit 20 $bin --batch $L/san3.txt --outdir /tmp/cs_$v > $O/cs_${v}_${tool}_$j.log 2>&1 ;;
        photo) $CS --tool $tool --print-limit 20 $bin $(sed -n 1p $L/sizes.txt) /tmp/cs.avif > $O/cs_${v}_${tool}_$j.log 2>&1 ;;
        thumb) $CS --tool $tool --print-limit 20 $bin $(sed -n 1p $L/thumbs.txt) /tmp/cs.avif > $O/cs_${v}_${tool}_$j.log 2>&1 ;;
      esac
      rc=$?
      kern=$(grep -a -o -E 'in (void )?[A-Za-z_][A-Za-z0-9_:]*' $O/cs_${v}_${tool}_$j.log | sort | uniq -c | sort -rn | head -4 | tr '\n' ' ')
      echo "CS c$v $tool $j: exit $rc $(grep -a -E 'ERROR SUMMARY|RACECHECK SUMMARY' $O/cs_${v}_${tool}_$j.log | tr '\n' ' ') kernels: $kern"
    done
  done
done
echo RUN_DONE
