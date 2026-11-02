#!/bin/bash
# checks2.sh: --check (bit-exact against av1core) on crop / resize runs; the error paths and the batch list options on
# JPEGs; --small-preset on the output size.  ~/results/checks2.txt
B=~/cr/build/av1gpu; A=~/main/build/av1gpu; IM=~/im7/AppRun
R=~/results/checks2.txt; : > $R
W=~/work/c2; rm -rf $W; mkdir -p $W
big=$(python3 -c "
import glob; from PIL import Image
fs = sorted(glob.glob('$HOME/images/*.jpg'), key=lambda f: Image.open(f).size[0] * Image.open(f).size[1]); print(fs[-1])")
mid=$(python3 -c "
import glob; from PIL import Image
fs = sorted(glob.glob('$HOME/images/*.jpg'), key=lambda f: Image.open(f).size[0] * Image.open(f).size[1])
print(next(f for f in fs if 2.5e6 < Image.open(f).size[0] * Image.open(f).size[1] < 5e6))")
rgba=$(ls ~/data/rgba/set/ui_hud_1080p.png)
gray=$(ls ~/ll/jpeg/*gray*.jpg 2>/dev/null | head -1)
{
echo "big $big ($(identify -format %wx%h $big)), mid $mid ($(identify -format %wx%h $mid)), gray $gray"
chk() { local n=$1 f=$2; shift 2; local o=$W/$n.avif
  $B $f $o --check "$@" > $W/$n.log 2>&1; echo "check $n: rc $? [$*] $(grep -h 'check:' $W/$n.log | tr '\n' ' ')"; }
chk jpeg420_crop_resize $big --crop 2001x1501+333+777 --max-size 1000x1000
chk jpeg444_10_resize $big --yuv 444 --depth 10 --max-height 1500
chk jpeg420_10_odd_crop $big --depth 10 --crop 1001x777+3+1
chk jpeg_gray_resize ${gray:-$big} --max-size 200x200
chk jpeg_exif6 ~/work/match/rot6.jpg --max-size 1600x1000
chk png_rgba_resize $rgba --max-size 300x300
chk png_rgba_444_10_crop_resize $rgba --yuv 444 --depth 10 --crop 1001x701+1+1 --max-size 500x500
chk png_lossless_crop ~/ll/kodak/kodim05.png --lossless --crop 301x201+1+1
chk png_lossless_ycgco_resize ~/ll/kodak/kodim05.png --lossless --ycgco-re --max-size 300x300
chk png16_444_10_resize ~/ll/png16_10/k2_10in16.png --yuv 444 --depth 10 --max-size 333x333
# ---- errors: single image ----
for o in "--crop 100x100+99999+0" "--crop 0x100+0+0" "--crop 100x100" "--crop 1,2,3" "--max-size 0x5" "--max-size abc" "--max-height 0" \
         "--max-mp 0" "--max-input-mp 2000"; do
  $B $big $W/err.avif $o > $W/err.log 2>&1; echo "error [$o]: rc $?: $(head -1 $W/err.log)"
done
# ---- batch list on JPEGs: options, a bad option, a crop outside the image, a TAB in a file name ----
mkdir -p $W/in; cp $big $W/in/a.jpg; cp $mid $W/in/b.jpg; cp $mid $W/in/c.jpg; cp $mid $W/in/d.jpg; cp $mid "$W/in/tab	name.jpg"
printf '%s\tcrop=1000x800+11+13\n%s\tmax-size=x300\tcrop=7,5,1200,900\n%s\tbogus=1\n%s\tcrop=10x10+50000+0\n%s\n' \
  $W/in/a.jpg $W/in/b.jpg $W/in/c.jpg $W/in/d.jpg "$W/in/tab	name.jpg" > $W/list.txt
$B --batch $W/list.txt --outdir $W/out > $W/list.log 2>&1; echo "batch list: rc $?"
grep -h "skipped\|images (" $W/list.log | sed 's/^/  /'
for f in $W/out/*.avif; do echo "  $(basename "$f"): $(python3 -c "
import struct,sys; d=open(sys.argv[1],'rb').read(); i=d.find(b'ispe'); print('%dx%d' % struct.unpack('>II', d[i+8:i+16]))" "$f")"; done
echo "  want: a.jpg 1000x800, b.jpg $(python3 -c "import math; s=300/900; print(math.floor(1200*s+0.5), 300)"), tab name: $(identify -format %wx%h $mid)"
# ---- --small-preset follows the output size: a 2.5-5 MP photo shrunk below 1 MP ----
mkdir -p $W/sp; cp $mid $W/sp/; cp $big $W/sp/
sp() { local n=$1; shift; rm -rf $W/sp_$n; $B --batch $W/sp --outdir $W/sp_$n "$@" > $W/sp_$n.log 2>&1; }
sp default --max-size 800x800
sp base --max-size 800x800 --preset base --small-preset none
sp turbo --max-size 800x800 --small-preset none
for f in $(ls $W/sp); do
  echo "small preset, $f shrunk to fit 800x800: default == --preset base --small-preset none: $(cmp -s $W/sp_default/$f.avif $W/sp_base/$f.avif && echo yes || echo NO);" \
       "default == --small-preset none (turbo): $(cmp -s $W/sp_default/$f.avif $W/sp_turbo/$f.avif && echo yes || echo no)"
done
sp big_default --max-size 2400x2400
sp big_turbo --max-size 2400x2400 --small-preset none
for f in $(ls $W/sp); do
  echo "small preset, $f shrunk to fit 2400x2400 (>= 1 MP): default == --small-preset none: $(cmp -s $W/sp_big_default/$f.avif $W/sp_big_turbo/$f.avif && echo yes || echo NO)"
done
} >> $R 2>&1
echo CHECKS2_DONE >> $R
