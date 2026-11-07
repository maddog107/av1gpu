#!/bin/bash
# targeted checks of the fix head's own changes
export T=/dev/shm/pfx; export BIN=${BIN:-$HOME/fix/build/av1gpu}; source ~/t/try.sh
export LD_LIBRARY_PATH=$HOME/im/lib; IM=~/im/bin/magick; L=~/adv/limits
echo "== a. the 65472-row refusal when the output has the source's size but a crop / resize still runs"
try $B $L/l64x65535.png $T/o.avif --crop 64x65000+0+0 --resize '64x65535!'
try $B $L/l64x65536.png $T/o.avif --crop 64x65000+0+0 --resize '64x65536!'
try $B $L/l64x65535.png $T/o.avif --crop 64x65000+0+0 --resize 'x65535'
rm -rf $T/b; printf '%s\n%s\tcrop=64x65000+0+0\tresize=64x65535!\n%s\n' ~/adv/trim/bar_mid_64x64.png $L/l64x65535.png ~/adv/trim/bar_mid_200x120.png > $T/l
$B --batch $T/l --outdir $T/b > /dev/null 2> $T/e; echo "batch: status $? files $(ls $T/b | wc -l)/3: $(grep -v 'stage ms' $T/e | grep -E 'skipped|CUDA|rror' | head -2 | cut -c1-150)"
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
a=np.full((65535,64,3),255,np.uint8); a[100:65000,10:50]=(10,90,200); a[0,:]=(255,255,255); wpng('/dev/shm/pfx/b65535.png',a)
PY
try $B $T/b65535.png $T/o.avif --trim --resize '64x65535!'
rm -rf $T/b; printf '%s\n%s\n' ~/adv/trim/bar_mid_64x64.png $T/b65535.png > $T/l; $B --batch $T/l --outdir $T/b --trim --resize '64x65535!' > /dev/null 2> $T/e; echo "batch --trim: status $? files $(ls $T/b | wc -l)/2: $(grep -v 'stage ms' $T/e | grep -E 'skipped|CUDA|rror' | head -2 | cut -c1-150)"
echo "== b. sources of 65536 rows (and 70000 with a crop) with tone and gray"
for o in "--negate" "--yuv 444 --negate" "--lossless --negate" "--yuv 444 --auto-level" "--auto-level" "--grayscale Average" "--yuv 444 --grayscale Average" "--lossless --grayscale RMS" "--depth 10 --gamma 2" "--yuv 444 --depth 10 --level 10%,90%" "--trim --fuzz 5%" "--yuv 444 --trim --fuzz 5% --negate"; do
  try $B $L/l64x65536.png $T/o.avif --preset turbo3 $o; done
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
a=np.full((70000,200,3),255,np.uint8); a[100:60000,20:180]=(30,120,200); wpng('/dev/shm/pfx/tall70k.png',a)
PY
for o in "--negate" "--yuv 444 --negate" "--lossless --negate" "--yuv 444 --auto-level" "--grayscale Average" "--trim"; do try $B $T/tall70k.png $T/o.avif --preset turbo3 --crop 200x60000+0+0 $o; done
try $B $T/tall70k.png $T/o.avif --preset turbo3
echo "== c. --negate exact at 4:2:0 (dump: Y' = 255 - Y, chroma' = 256 - C, clipped)"
python3 - <<'PY'
import subprocess, os, sys, glob, numpy as np
B=os.environ['BIN']; T='/dev/shm/pfx'
for f in sorted(glob.glob(os.path.expanduser('~/images_std/*.jpg')))[:4]+[os.path.expanduser('~/adv/trim/noise_257x65.png'),os.path.expanduser('~/tone2/lineart.png'),os.path.expanduser('~/adv/jpeg/s411_161x97.jpg')]:
    r=[]
    for o in ([],['--negate'],['--depth','10'],['--depth','10','--negate']):
        subprocess.run([B,f,f'{T}/o.avif','--dump-source',f'{T}/d.yuv','--preset','turbo3']+o,check=True,capture_output=True)
        r.append(open(f'{T}/d.yuv','rb').read())
    from PIL import Image
    w,h=Image.open(f).size; n=w*h; c=((w+1)//2)*((h+1)//2)
    for k,(a0,a1,top,dt) in enumerate(((r[0],r[1],255,np.uint8),(r[2],r[3],1023,'<u2'))):
        a0=np.frombuffer(a0,dt).astype(int); a1=np.frombuffer(a1,dt).astype(int)
        y=np.array_equal(a1[:n],top-a0[:n]); mid=2*(128 if top==255 else 512)
        cexp=np.clip(mid-a0[n:n+2*c],0,top); ce=np.abs(a1[n:n+2*c]-cexp)
        print(os.path.basename(f)[:28], '10-bit' if k else '8-bit', 'Y exact' if y else f'Y differs at {(a1[:n]!=top-a0[:n]).sum()} px', f'chroma max diff {ce.max()} ({(ce>0).sum()} samples)')
PY
echo "== d. --max-mp hint: is the value given enough?"
python3 - <<'PY'
from PIL import Image
import numpy as np
for w,h in ((6001,6001),(7000,7000)):
    a=np.full((h,w,3),255,np.uint8); a[100:200,100:200]=0; Image.fromarray(a).save(f'/dev/shm/pfx/big{w}.jpg',quality=80)
PY
for w in 6001 7000; do f=$T/big$w.jpg; try $B $f $T/o.avif; n=$(grep -o 'max-mp [0-9]*' $T/err.txt | tail -1 | cut -d' ' -f2); echo "  hint: --max-mp $n"; [ -n "$n" ] && try $B $f $T/o.avif --max-mp $n
  rm -rf $T/b; echo $f > $T/l; $B --batch $T/l --outdir $T/b > /dev/null 2> $T/e; n=$(grep -o 'max-mp [0-9]*' $T/e | tail -1 | cut -d' ' -f2); echo "  batch hint --max-mp $n: $(grep skipped $T/e | cut -c1-140)"; rm -rf $T/b; $B --batch $T/l --outdir $T/b --max-mp $n > /dev/null 2>&1; echo "  with it: status $? files $(ls $T/b | wc -l)"
  try $B $f $T/o.avif --resize 200%; try $B $f $T/o.avif --trim --fuzz 5% --resize 7000x7000; done
try $B $T/big7000.jpg $T/o.avif --max-mp 2000
echo "== e. two-tone pages, --level with one %, --colorspace case"
bash ~/t/two.sh 2>&1 | grep -E 'two_|batch' | cut -c1-170
for g in "10,90%" "10%,90" "10%,90%" "10,90"; do try $B ~/adv/trim/hgrad_256x64.png $T/o.avif --lossless --level $g; done
try $B ~/adv/trim/hgrad_256x64.png $T/o.avif --colorspace GRAY
echo "== f. --help on stdout, status"
$B --help > $T/h.txt 2> $T/he.txt; echo "--help: status $? stdout $(wc -l < $T/h.txt) lines, stderr $(wc -c < $T/he.txt) bytes"; $B > /dev/null 2> $T/he.txt; echo "no arguments: status $? stderr $(wc -l < $T/he.txt) lines"
echo PROBEFIX_DONE
