#!/bin/bash
# san.sh: compute-sanitizer memcheck (--leak-check full) on crop + resize runs of the branch, in parallel; every output
# must equal the uninstrumented run's.  Sets: 3 size-range photos (as sanitizer.sh picks them: >= 2 MP with 4x4
# tiles, 1-2 MP, < 1 MP), 6 RGBA PNGs, and the 82 MP JPEG (CPU-route decode, --max-input-mp).  ~/results/san.txt,
# logs in ~/results/san/.
B=~/cr/build/av1gpu; CS=/usr/local/cuda/bin/compute-sanitizer
D=~/sets/san3; rm -rf $D; mkdir -p $D ~/sets/san_rgba ~/sets/san_big
python3 - <<'EOF'
import glob, os
from PIL import Image
fs = sorted(glob.glob(os.path.expanduser("~/images/*.jpg")))
mp = {f: Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in fs}
pick = [min((f for f in fs if mp[f] >= 2), key=mp.get), min((f for f in fs if 1 <= mp[f] < 2), key=lambda f: abs(mp[f] - 1.5)),
        min((f for f in fs if 0.5 <= mp[f] < 1), key=lambda f: abs(mp[f] - 0.75))]
for f in pick:
    os.symlink(f, os.path.expanduser("~/sets/san3/") + os.path.basename(f))
    print(os.path.basename(f), Image.open(f).size, round(mp[f], 2), "MP")
rg = [f for f in sorted(glob.glob(os.path.expanduser("~/data/rgba/set/*.png"))) if Image.open(f).mode in ("RGBA", "LA")][:6]
for f in rg:
    d = os.path.expanduser("~/sets/san_rgba/") + os.path.basename(f)
    if not os.path.exists(d):
        os.symlink(f, d)
EOF
ln -sf ~/sets/large/b82_420.jpg ~/sets/san_big/
L=~/results/san; rm -rf $L; mkdir -p $L ~/work/san
R=~/results/san.txt
{ echo "BIN $B ($(cat ~/cr/COMMIT))"; ls ~/sets/san3 ~/sets/san_rgba ~/sets/san_big; $CS --version | tail -1; } > $R
run() { local n=$1 src=$2; shift 2; local o=~/work/san/$n
  rm -rf $o.ref $o.mc; $B --batch $src --outdir $o.ref "$@" > $L/$n.ref.log 2>&1
  local t=$(date +%s)
  timeout 3600 $CS --tool memcheck --leak-check full --print-limit 100 --error-exitcode 99 $B --batch $src --outdir $o.mc "$@" > $L/$n.log 2>&1
  local rc=$? d=0
  for f in $o.ref/*; do cmp -s $f $o.mc/$(basename $f) || d=$((d + 1)); done
  echo "memcheck $n [$*]: rc $rc, $(( $(date +%s) - t )) s, outputs $(ls $o.mc 2>/dev/null | wc -l), differing from the plain run: $d; $(grep -E 'SUMMARY' $L/$n.log | tail -1)" > $L/$n.res; }
run photos_420_8 $D --crop 65536x65536+1+1 --max-size 1201x997 &
run photos_444_10 $D --yuv 444 --depth 10 --crop 65536x65536+3+1 --max-size 999x999 &
run rgba_420_8 ~/sets/san_rgba --crop 65536x65536+1+3 --max-size 257x257 &
run rgba_444_10 ~/sets/san_rgba --yuv 444 --depth 10 --max-size 301x301 &
run big82_shrink ~/sets/san_big --max-input-mp 100 --max-height 1500 &
wait
# single-image mode (stream 0, cudaFree) on one photo
f=$(ls $D | head -1); t=$(date +%s)
$B $D/$f ~/work/san/single.ref.avif --crop 65536x65536+1+1 --max-size 1201x997 > $L/single.ref.log 2>&1
timeout 3600 $CS --tool memcheck --leak-check full --error-exitcode 99 $B $D/$f ~/work/san/single.mc.avif --crop 65536x65536+1+1 --max-size 1201x997 > $L/single.log 2>&1
echo "memcheck single [$f --crop 65536x65536+1+1 --max-size 1201x997]: rc $?, $(( $(date +%s) - t )) s, same as plain: $(cmp -s ~/work/san/single.ref.avif ~/work/san/single.mc.avif && echo yes || echo NO); $(grep SUMMARY $L/single.log | tail -1)" > $L/single.res
cat $L/*.res >> $R
echo SAN_DONE >> $R
