#!/bin/bash
# the host / kernels / contrarian items on the real binaries
export T=/dev/shm/r2; source ~/t/try.sh
export LD_LIBRARY_PATH=$HOME/im/lib; IM=~/im/bin/magick
echo "=== host 1 / G1: EXIF 6 JPEG stored 4032x3024"
python3 - <<'PY'
from PIL import Image
import numpy as np
a=np.zeros((3024,4032,3),np.uint8); a[...,0]=np.linspace(0,255,4032).astype(np.uint8); a[...,1]=np.linspace(0,255,3024).astype(np.uint8)[:,None]; a[500:1500,300:1800]=(200,30,30)
ex=Image.Exif(); ex[0x0112]=6
Image.fromarray(a).save('/dev/shm/r2/phone.jpg',quality=88,exif=ex)
ex=Image.Exif(); ex[0x0112]=1
Image.fromarray(a).save('/dev/shm/r2/flat.jpg',quality=88,exif=ex)
b=np.full((700,1000,3),255,np.uint8); b[100:600,200:800]=(10,120,60)
Image.fromarray(b).save('/dev/shm/r2/any.png')
PY
for g in 50% 150% 50%x x50%; do
  $IM $T/phone.jpg -auto-orient -resize $g -format "magick -auto-orient -resize $g: displayed %wx%h\n" info:
  try $B $T/phone.jpg $T/o.avif --resize $g --max-mp 100
done
rm -rf $T/b; mkdir $T/b; printf '%s\n' $T/phone.jpg $T/flat.jpg > $T/l; $B --batch $T/l --outdir $T/b --resize 50% > /dev/null 2>$T/e; echo "batch --resize 50%: rc=$?"
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in sorted(glob.glob('$T/b/*.avif')):
    c=avt.check(f); print(' ',f.split('/')[-1], 'stored', c['si'][1:3], 'irot',c['irot'],'decodes in dav1d and aomdec, same pixels')
PY
echo "=== host 1 / G2: x50%, any image"
for g in x50% x150%; do
  $IM $T/any.png -resize $g -format "magick $g: %wx%h\n" info:; try $B $T/any.png $T/o.avif --resize $g
done
rm -rf $T/b; echo $T/any.png > $T/l; $B --batch $T/l --outdir $T/b --resize x50% > /dev/null 2>$T/e; echo "batch --resize x50%: rc=$?"
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in sorted(glob.glob('$T/b/*.avif')):
    c=avt.check(f); print(' ',f.split('/')[-1], 'stored', c['si'][1:3])
PY
echo "=== host 2 / G3: --resize 70000x64!"
for f in $T/any.png ~/adv/jpeg/ex0_420_160x96.jpg; do
  rm -f $T/o.avif*; timeout 120 $B $f $T/o.avif --debug-outputs --resize '70000x64!' > /dev/null 2> $T/e; rc=$?; echo "$(basename $f): status $rc, $(ls -la $T/o.avif 2>/dev/null | awk '{print $5}') bytes"
  dav1d -q -i $T/o.avif.obu --demuxer section5 -o /dev/null --muxer null > $T/d 2>&1; echo "  dav1d on .obu: status $? $(tail -2 $T/d | tr '\n' ' ' | cut -c1-120)"
  aomdec --rawvideo -o /dev/null $T/o.avif.obu > $T/d 2>&1; echo "  aomdec on .obu: status $? $(tail -1 $T/d | cut -c1-120)"
  avifdec $T/o.avif $T/o.png > $T/d 2>&1; echo "  avifdec on .avif: status $? $(tail -1 $T/d | cut -c1-120)"
done
echo "--- with an alpha item, and past the tile limit"
try $B ~/adv/alpha/hidden_120x80.png $T/o.avif --resize '70000x64!'
try $B $T/any.png $T/o.avif --resize '300000x1!'
rm -rf $T/b; echo $T/any.png > $T/l; $B --batch $T/l --outdir $T/b --resize '70000x64!' > /dev/null 2>$T/e; echo "batch: status $? files $(ls $T/b | wc -l): $(grep -v 'stage ms' $T/e | tail -1 | cut -c1-120)"
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in sorted(glob.glob('$T/b/*.avif')):
    try: c=avt.check(f); print('  decodes', c['si'])
    except avt.Bad as e: print('  BROKEN:', str(e).split(chr(10))[0])
PY
echo "=== G3 pre-existing: a PNG wider than 65536 on main, no new options"
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
for w in (65536,65537,70000):
    a=np.full((64,w,3),255,np.uint8); a[20:40,1000:2000]=(10,20,200); wpng(f'/dev/shm/r2/w{w}.png',a)
    al=np.full((64,w),255,np.uint8); al[:8]=0; wpng(f'/dev/shm/r2/wa{w}.png',np.dstack([a,al]))
PY
for f in w65536 w65537 w70000 wa65537 wa70000; do try $M $T/$f.png $T/o.avif; done
rm -f $T/o.avif*; $M $T/w70000.png $T/o.avif --debug-outputs > /dev/null 2>&1; echo "main w70000 status $?"; dav1d -q -i $T/o.avif.obu --demuxer section5 -o /dev/null --muxer null 2>&1 | tail -1 | cut -c1-100; aomdec --rawvideo -o /dev/null $T/o.avif.obu 2>&1 | tail -1 | cut -c1-100
echo "=== contrarian 7: --level with a single %"
python3 - <<'PY'
import subprocess, os, sys, numpy as np
from PIL import Image
sys.path.insert(0,'/home/ubuntu/t'); import avt
T='/dev/shm/r2'; M=os.path.expanduser('~/im/bin/magick'); B=os.environ.get('BIN',os.path.expanduser('~/pr/build/av1gpu')); env=dict(os.environ,LD_LIBRARY_PATH=os.path.expanduser('~/im/lib'))
src=os.path.expanduser('~/adv/trim/hgrad_256x64.png')
for g in ('10%,90%','10,90%','10%,90','10,90'):
    r=subprocess.run([M,src,'-level',g,'-depth','8',f'PNG24:{T}/m.png'],capture_output=True,text=True,env=env)
    a=np.array(Image.open(f'{T}/m.png').convert('RGB')).astype(int)
    p=subprocess.run([B,src,f'{T}/q.avif','--lossless','--level',g],capture_output=True,text=True)
    if p.returncode: print(f"--level {g}: av1gpu status {p.returncode}: {[l for l in p.stderr.splitlines() if 'bad' in l][:1]}; magick output range {a.min()}..{a.max()}"); continue
    c=avt.check(f'{T}/q.avif'); d=np.rint(avt.to_rgb(c['si'],c['planes'])*255).astype(int)
    print(f"--level {g}: mean |av1gpu - magick| {np.abs(d-a).mean():.2f}, max {np.abs(d-a).max()}; magick range {a.min()}..{a.max()}, av1gpu range {d.min()}..{d.max()}")
PY
echo "=== contrarian's own script"
bash ~/repro_for_aws.sh $B $IM 2>&1 | cut -c1-150
echo ROUND2_DONE
