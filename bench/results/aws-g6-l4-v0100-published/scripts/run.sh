#!/bin/bash
# rc12: the published v0.10.0 Linux packages (B, from the release page, built from the tag) against v0.9.0 (A): rc11's
# byte-identity runs and cold single-image start-up (median of 12, binaries alternating, v0.9.0 twice for noise).
O=~/out; mkdir -p $O ~/bin ~/rel ~/src ~/img; exec > $O/run.log 2>&1
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q >/dev/null; sudo apt-get install -yq python3-pil python3-numpy bc >/dev/null 2>&1
tar xzf ~/src.tar.gz -C ~/src
python3 ~/src/bench/fetch_images.py --manifest ~/src/bench/openimages_v4_gamut.csv --out ~/img/sizes | tail -1
python3 ~/src/bench/fetch_images.py --out ~/img/std | tail -1
python3 - <<'PY'
# thumbnails as production makes them; grayscale JPEGs; RGBA PNGs with a radial alpha; small RGB PNGs for --lossless;
# the single-image start-up inputs (as the v0.9.0 check: the photo nearest 2.5 MP in 2-3 MP, nearest 11 MP in 10-12)
import glob, os
import numpy as np
from PIL import Image, ImageOps
H = os.path.expanduser("~/img")
for d in ("thumbs", "mono", "alpha", "ll", "st"): os.makedirs(f"{H}/{d}", exist_ok=True)
fs = sorted(glob.glob(f"{H}/sizes/*.jpg"))
mp = {}
for i, f in enumerate(fs):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB"); b = os.path.basename(f)[:-4]
    mp[f] = im.size[0] * im.size[1] / 1e6
    ImageOps.fit(im, (512, 384), Image.LANCZOS).save(f"{H}/thumbs/d512_{b}.jpg", quality=85)
    t = im.copy(); t.thumbnail((320, 320), Image.LANCZOS); t.save(f"{H}/thumbs/t320_{b}.jpg", quality=85)
    if i % 5 == 0 and mp[f] <= 12:
        im.convert("L").save(f"{H}/mono/m_{b}.jpg", quality=90)
    if i % 10 == 1:
        a = im.copy(); a.thumbnail((1600, 1600), Image.LANCZOS)
        w, h = a.size; y, x = np.mgrid[0:h, 0:w]
        al = np.clip(255 * (1.3 - np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2))), 0, 255).astype(np.uint8)
        a.putalpha(Image.fromarray(al)); a.save(f"{H}/alpha/a_{b}.png")
    if i % 10 == 2:
        im.resize((im.size[0] // 8 or 1, im.size[1] // 8 or 1), Image.LANCZOS).save(f"{H}/ll/l_{b}.png")
photo = min((f for f in fs if 2 <= mp[f] <= 3), key=lambda f: abs(mp[f] - 2.5))
big = min((f for f in fs if 10 <= mp[f] <= 12), key=lambda f: abs(mp[f] - 11))
for n, f in (("photo", photo), ("big", big)):
    os.system(f"cp {f} {H}/st/{n}.jpg"); print(n, os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
b = os.path.basename(photo)[:-4]
os.system(f"cp {H}/thumbs/d512_{b}.jpg {H}/st/d512.jpg; cp {H}/thumbs/t320_{b}.jpg {H}/st/t320.jpg")
m = sorted(glob.glob(f"{H}/mono/*.jpg"), key=lambda f: abs(Image.open(f).size[0] * Image.open(f).size[1] / 1e6 - 2.5))[0]
a = sorted(glob.glob(f"{H}/alpha/*.png"))[0]
os.system(f"cp {m} {H}/st/mono.jpg; cp {a} {H}/st/alpha.png")
for f in sorted(glob.glob(f"{H}/st/*")): print(os.path.basename(f), Image.open(f).size, Image.open(f).mode)
PY
L=$O/lists; mkdir -p $L
for d in sizes std thumbs mono alpha ll; do ls ~/img/$d/* > $L/$d.txt; done
head -10 $L/sizes.txt > $L/mixed.txt; cat $L/mono.txt >> $L/mixed.txt
{ head -12 $L/sizes.txt; head -6 $L/thumbs.txt; head -4 $L/mono.txt; head -4 $L/alpha.txt; } > $L/single.txt
{ cat $L/ll.txt; head -4 $L/alpha.txt; } > $L/single_ll.txt
wc -l $L/*.txt; sha256sum ~/img/*/* > $O/inputs.sha256
echo PREP_DONE
mkdir -p ~/pkg
# the v0.10.0 assets appear when the tag's Release run publishes (up to ~90 min after the tag)
for i in $(seq 120); do curl -fsSI https://github.com/maddog107/av1gpu/releases/download/v0.10.0/av1gpu-0.10.0-linux-x86_64-cuda12.tar.gz.sha256 >/dev/null && break; sleep 60; done
for v in 0.9.0 0.10.0; do
  d=~/rel; [ $v = 0.10.0 ] && d=~/pkg; cd $d; G=https://github.com/maddog107/av1gpu/releases/download/v$v
  for s in "" -cuda12; do f=av1gpu-$v-linux-x86_64$s.tar.gz
    curl -fsSLO $G/$f && curl -fsSLO $G/$f.sha256 && sha256sum -c $f.sha256 && tar xzf $f || { echo REL_FAIL $v; exit 1; }; done
done
cd ~
sha256sum ~/rel/*.tar.gz ~/pkg/*.tar.gz
A=~/rel/av1gpu-0.9.0-linux-x86_64/av1gpu; A12=~/rel/av1gpu-0.9.0-linux-x86_64-cuda12/av1gpu
B=~/pkg/av1gpu-0.10.0-linux-x86_64/av1gpu; B12=~/pkg/av1gpu-0.10.0-linux-x86_64-cuda12/av1gpu
cp $A ~/bin/Ac   # v0.9.0 from a second path: the noise of the timing
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader; uname -r; nproc
for b in $A $B $A12 $B12; do echo "$b: $($b --version 2>&1 | head -1)"; ls -la $b; done
[ -x $B ] && [ -x $B12 ] || { echo PKG_MISSING; exit 1; }
TOT=0; IDN=0
pair() {   # name binA binB list args...: both binaries in batch mode on the list, cmp every output and the exit status
  local n=$1 a=$2 b=$3 l=$4; shift 4; local side bin
  for side in A B; do
    bin=$a; [ $side = B ] && bin=$b; mkdir -p $O/r/$n/$side
    $bin --batch $l --outdir $O/r/$n/$side "$@" > $O/r/$n/$side.log 2>&1; echo $? > $O/r/$n/$side.rc
  done
  local tot=0 d=0 f
  for f in $O/r/$n/A/*.avif; do tot=$((tot + 1)); cmp -s $f $O/r/$n/B/$(basename $f) || { d=$((d + 1)); echo "  DIFF $n $(basename $f)"; }; done
  local na=$(ls $O/r/$n/A | wc -l) nb=$(ls $O/r/$n/B | wc -l)
  [ $na = $nb ] || { d=$((d + 1)); echo "  COUNT MISMATCH $n $na $nb"; }
  [ "$(cat $O/r/$n/A.rc)" = "$(cat $O/r/$n/B.rc)" ] || { d=$((d + 1)); echo "  EXIT MISMATCH $n"; }
  TOT=$((TOT + tot)); IDN=$((IDN + tot - d))
  echo "PAIR $n: $((tot - d))/$tot identical (exit A $(cat $O/r/$n/A.rc) B $(cat $O/r/$n/B.rc); list $(wc -l < $l); GP_ENCODERS=${GP_ENCODERS:-default})"
}
single() {   # name binA binB list args...: one process per file, cmp every output and exit status
  local n=$1 a=$2 b=$3 l=$4; shift 4; local tot=0 d=0 f o ra rb
  mkdir -p $O/r/$n/A $O/r/$n/B
  for f in $(cat $l); do
    o=$(basename ${f%.*}).avif
    $a "$@" $f $O/r/$n/A/$o > /dev/null 2>&1; ra=$?; $b "$@" $f $O/r/$n/B/$o > /dev/null 2>&1; rb=$?
    tot=$((tot + 1)); if [ $ra = $rb ] && cmp -s $O/r/$n/A/$o $O/r/$n/B/$o; then :; else d=$((d + 1)); echo "  DIFF $n $o (exit $ra $rb)"; fi
  done
  TOT=$((TOT + tot)); IDN=$((IDN + tot - d))
  echo "SINGLE $n: $((tot - d))/$tot identical"
}
repeat() {   # name k ref args... in out: k processes of B on one input, each output cmp'd with A's (ref)
  local n=$1 k=$2 ref=$3; shift 3; local i d=0 out=$O/r/rep_$n.avif
  for i in $(seq $k); do rm -f $out; $B "$@" $out > /dev/null 2>&1; cmp -s $out $ref || { d=$((d + 1)); echo "  DIFF repeat $n run $i"; }; done
  TOT=$((TOT + k)); IDN=$((IDN + k - d)); echo "REPEAT $n: $((k - d))/$k identical to v0.9.0"
}
echo "== 1. batch mode, CUDA 13.4 packages"
pair sizes $A $B $L/sizes.txt
pair sizes_444 $A $B $L/sizes.txt --yuv 444
pair sizes_10 $A $B $L/sizes.txt --depth 10
pair sizes_444_10 $A $B $L/sizes.txt --yuv 444 --depth 10
pair std $A $B $L/std.txt
pair thumbs $A $B $L/thumbs.txt
pair mono $A $B $L/mono.txt
pair mono_444 $A $B $L/mono.txt --yuv 444
export GP_ENCODERS=4; pair mono_444_enc4 $A $B $L/mono.txt --yuv 444; pair mono_444_10_enc4 $A $B $L/mono.txt --yuv 444 --depth 10; unset GP_ENCODERS
pair mixed_444 $A $B $L/mixed.txt --yuv 444
pair alpha $A $B $L/alpha.txt
pair alpha_444 $A $B $L/alpha.txt --yuv 444
pair alpha_444_10 $A $B $L/alpha.txt --yuv 444 --depth 10
pair ll $A $B $L/ll.txt --lossless
pair ll_alpha $A $B $L/alpha.txt --lossless
echo "SUBTOTAL batch: $IDN/$TOT"; T1=$TOT; I1=$IDN
echo "== 2. single-file mode, CUDA 13.4 packages"
single s420 $A $B $L/single.txt
single s444 $A $B $L/single.txt --yuv 444
single s10 $A $B $L/single.txt --depth 10
single s444_10 $A $B $L/single.txt --yuv 444 --depth 10
single sll $A $B $L/single_ll.txt --lossless
$A ~/img/st/photo.jpg $O/r/ref_photo.avif; $A --yuv 444 ~/img/st/mono.jpg $O/r/ref_mono444.avif; $A --yuv 444 ~/img/st/alpha.png $O/r/ref_alpha444.avif
repeat photo 20 $O/r/ref_photo.avif ~/img/st/photo.jpg
repeat mono_444 10 $O/r/ref_mono444.avif --yuv 444 ~/img/st/mono.jpg
repeat alpha_444 10 $O/r/ref_alpha444.avif --yuv 444 ~/img/st/alpha.png
echo "SUBTOTAL single: $((IDN - I1))/$((TOT - T1))"; T2=$TOT; I2=$IDN
echo "== 3. CUDA 12.9 packages"
single c12_s420 $A12 $B12 $L/single.txt
single c12_s444_10 $A12 $B12 $L/single.txt --yuv 444 --depth 10
pair c12_sizes $A12 $B12 $L/sizes.txt
echo "SUBTOTAL c12: $((IDN - I2))/$((TOT - T2))"
echo "TOTAL: $IDN/$TOT identical"
echo "== 4. cold start, one process per run, median of 12 (A = v0.9.0, Ac = v0.9.0 again, B = v0.10.0)"
W="python3 $HOME/wall9.py 12 $A $HOME/bin/Ac $B --"
$W ~/img/st/photo.jpg /tmp/w{bin}.avif
$W ~/img/st/big.jpg /tmp/w{bin}.avif
$W ~/img/st/d512.jpg /tmp/w{bin}.avif
$W ~/img/st/t320.jpg /tmp/w{bin}.avif
$W --yuv 444 --depth 10 ~/img/st/photo.jpg /tmp/w{bin}.avif
$W --yuv 444 ~/img/st/mono.jpg /tmp/w{bin}.avif
$W ~/img/st/alpha.png /tmp/w{bin}.avif
printf '%s\n' ~/img/st/photo.jpg ~/img/st/d512.jpg > $L/pair.txt
$W --batch $L/pair.txt --outdir /tmp/wb{bin}
echo RUN_DONE
