#!/bin/bash
# large.sh: pictures above 35.6 MP made with ImageMagick from the largest size-range photo: 5587x7365 (41 MP) and
# 11174x7365 (82 MP, two side by side) 4:2:0 JPEGs (q90), a 4:4:4 one, and a 41 MP RGB PNG.  Without the options they
# are skipped as on main; --max-mp 100 encodes them (level, GPU memory peak, host RSS; decoder gate; --check);
# --max-input-mp 100 --max-height 2960 decodes them at full size and shrinks them.  ~/results/large.txt
B=~/cr/build/av1gpu; A=~/main/build/av1gpu; IM=~/im7/AppRun; G=~/cr/tests/decoder_gate.py
R=~/results/large.txt; : > $R
L=~/sets/large; W=~/work/large; mkdir -p $L $W
big=$(python3 -c "
import glob; from PIL import Image
fs = sorted(glob.glob('$HOME/images/*.jpg'), key=lambda f: Image.open(f).size[0] * Image.open(f).size[1]); print(fs[-1])")
if [ ! -e $L/b82_420.jpg ]; then
  $IM $big -resize '5587x7365!' -quality 90 -sampling-factor 2x2 $L/b41_420.jpg
  $IM $big -flop -resize '5587x7365!' -quality 90 -sampling-factor 2x2 $L/b41b_420.jpg
  $IM $L/b41_420.jpg $L/b41b_420.jpg +append -quality 90 -sampling-factor 2x2 $L/b82_420.jpg
  $IM $L/b41_420.jpg -quality 92 -sampling-factor 1x1 $L/b41_444.jpg
  $IM $L/b41b_420.jpg $L/b41_png.png
fi
av1c() { python3 -c "
import struct, sys
d = open(sys.argv[1], 'rb').read(); i = d.find(b'ispe'); j = d.find(b'av1C')
print('%dx%d level %d' % (struct.unpack('>II', d[i + 8:i + 16]) + (d[j + 5] & 31,)))" "$1"; }
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > $W/mem_$1.txt & echo $!; }
peak() { sort -n $W/mem_$1.txt | tail -1; }
{
ls -l $L | tail -n +2 | awk '{print $5, $9}'; for f in $L/*; do echo "$(basename $f): $(identify -format '%wx%h %[jpeg:sampling-factor]' $f 2>/dev/null)"; done
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); echo "GPU memory used before the runs: $base MiB"
# ---- 1. no options: skipped as on main ----
for b in A B; do bin=${!b}; rm -rf $W/no_$b; $bin --batch $L --outdir $W/no_$b > $W/no_$b.log 2>&1; echo "no options $b: rc $?, outputs $(ls $W/no_$b 2>/dev/null | wc -l)"; done
diff <(grep -h skipped $W/no_A.log) <(grep -h skipped $W/no_B.log) > /dev/null && echo "no options: skip lines identical to main:" || echo "no options: skip lines DIFFER"
grep -h skipped $W/no_B.log | sed 's/^/  /'
for f in b41_420.jpg b41_png.png; do
  for b in A B; do bin=${!b}; $bin $L/$f $W/s_$b.avif > $W/s_$b.log 2>&1; echo "single $f $b: rc $?: $(head -1 $W/s_$b.log)"; done
  [ -e $W/s_A.avif ] && { cmp -s $W/s_A.avif $W/s_B.avif && echo "  single $f: identical to main" || echo "  single $f: DIFFERS from main"; }
  rm -f $W/s_A.avif $W/s_B.avif
done
# ---- 2. --max-mp 100: encoded at full size ----
rm -rf $W/mp100; m=$(mem mp100)
/usr/bin/time -v $B --batch $L --outdir $W/mp100 --max-mp 100 --stats > $W/mp100.json 2> $W/mp100.log; rc=$?; kill $m
echo "--max-mp 100: rc $rc, GPU memory peak $(peak mp100) MiB (process: peak minus $base), host max RSS $(grep 'Maximum resident' $W/mp100.log | awk '{print $NF}') kB"
grep -h "images (\|peak dev" $W/mp100.log | sed 's/^/  /'
for f in $W/mp100/*.avif; do echo "  $(basename $f): $(av1c $f)"; done
for f in b41_420.jpg b82_420.jpg; do rm -rf $W/one; m=$(mem one_$f)
  $B --batch <(echo $L/$f) --outdir $W/one --max-mp 100 --stats > $W/one.json 2> $W/one.log; kill $m
  echo "  $f alone: GPU memory peak $(peak one_$f) MiB; $(grep -h 'images (' $W/one.log | cut -c1-90)"; done
python3 $G --bin $B --jobs 8 --images $L --q 120 -- --max-mp 100 > $W/gate.log 2>&1; echo "gate --max-mp 100: rc $?"; tail -n 1 $W/gate.log | sed 's/^/  /'
python3 $G --bin $B --jobs 8 --images $L --q 120 --expect-format 444 --expect-depth 10 -- --max-mp 100 --yuv 444 --depth 10 > $W/gate10.log 2>&1
echo "gate --max-mp 100 --yuv 444 --depth 10: rc $?"; tail -n 1 $W/gate10.log | sed 's/^/  /'
for f in b41_420.jpg b82_420.jpg b41_png.png; do $B $L/$f $W/chk.avif --max-mp 100 --check > $W/chk.log 2>&1; echo "--check $f --max-mp 100: rc $? $(grep -h 'check:' $W/chk.log)"; done
# ---- 3. --max-input-mp 100 --max-height 2960: decoded at full size, shrunk ----
rm -rf $W/shrink; m=$(mem shrink)
/usr/bin/time -v $B --batch $L --outdir $W/shrink --max-input-mp 100 --max-height 2960 --stats > $W/shrink.json 2> $W/shrink.log; rc=$?; kill $m
echo "--max-input-mp 100 --max-height 2960: rc $rc, GPU memory peak $(peak shrink) MiB, host max RSS $(grep 'Maximum resident' $W/shrink.log | awk '{print $NF}') kB"
grep -h "images (" $W/shrink.log | sed 's/^/  /'
for f in $W/shrink/*.avif; do echo "  $(basename $f): $(av1c $f)"; done
python3 $G --bin $B --jobs 8 --images $L --q 120 -- --max-input-mp 100 --max-height 2960 > $W/gates.log 2>&1; echo "gate --max-input-mp 100 --max-height 2960: rc $?"
tail -n 1 $W/gates.log | sed 's/^/  /'
# --max-input-mp without a crop / resize: decoded, then skipped (larger than --max-mp)
$B --batch $L --outdir $W/onlyin --max-input-mp 100 > $W/onlyin.log 2>&1; echo "--max-input-mp 100 alone: rc $?, outputs $(ls $W/onlyin 2>/dev/null | wc -l)"
grep -h skipped $W/onlyin.log | sed 's/^/  /'
} >> $R 2>&1
echo LARGE_DONE >> $R
