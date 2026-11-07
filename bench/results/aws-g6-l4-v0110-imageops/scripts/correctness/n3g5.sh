#!/bin/bash
# real ImageMagick on alpha (N3) and on the filter it picks (G5), then av1gpu for comparison
export LD_LIBRARY_PATH=$HOME/im/lib; M=~/im/bin/magick; B=${BIN:-$HOME/pr/build/av1gpu}; T=/dev/shm/n3; rm -rf $T; mkdir -p $T
python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, '/home/ubuntu/t')
from gen_lib import wpng, chunk
from PIL import Image
T='/dev/shm/n3'
h,w=80,120
c=np.random.default_rng(1).integers(0,256,(h,w,3),dtype=np.uint8)
a=np.tile(np.linspace(0,255,w).astype(np.uint8)[None,:],(h,1))
wpng(f'{T}/rgba.png',np.dstack([c,a]))
wpng(f'{T}/rgb.png',c)
wpng(f'{T}/gray.png',c[...,0])
idx=(c[...,0]>>6).astype(np.uint8)
pal=np.array([[255,255,255],[255,0,0],[0,255,0],[0,0,255]],np.uint8)
wpng(f'{T}/pal.png',idx,ct=3,extra=chunk(b'PLTE',pal.tobytes()))
Image.fromarray(c).save(f'{T}/c.jpg',quality=92); Image.fromarray(c[...,0]).save(f'{T}/g.jpg',quality=92)
PY
echo "=== N3: does each operation change alpha on an RGBA PNG (real magick)?"
python3 - <<'PY'
import subprocess, os, numpy as np
from PIL import Image
T='/dev/shm/n3'; M=os.path.expanduser('~/im/bin/magick'); env=dict(os.environ,LD_LIBRARY_PATH=os.path.expanduser('~/im/lib'))
src=np.array(Image.open(f'{T}/rgba.png'))
print(f"source alpha at x=0,30,60,90,119: {[int(src[0,x,3]) for x in (0,30,60,90,119)]}")
for ops in (['-negate'],['-channel','RGB','-negate','+channel'],['-level','10%,90%'],['-channel','RGB','-level','10%,90%','+channel'],['-gamma','2'],['-channel','RGB','-gamma','2','+channel'],['-auto-level'],['-channel','RGB','-auto-level','+channel'],['-grayscale','Rec709Luma'],['-colorspace','Gray']):
    subprocess.run([M,f'{T}/rgba.png']+ops+[f'PNG32:{T}/o.png'],check=True,env=env)
    o=np.array(Image.open(f'{T}/o.png').convert('RGBA'))
    same=np.array_equal(o[...,3],src[...,3]); inv=np.array_equal(o[...,3],255-src[...,3])
    print(f"{' '.join(ops):42s} alpha {'unchanged' if same else 'exactly inverted' if inv else 'changed'}; alpha at x=0,30,60,90,119: {[int(o[0,x,3]) for x in (0,30,60,90,119)]}; RGB under alpha 255 column changed: {int((o[:,-1,:3]!=src[:,-1,:3]).any())}")
PY
echo "--- the same via the kernels report's one-liner: magick rgba.png -negate -alpha extract a.png"
$M $T/rgba.png -negate -alpha extract $T/a.png; $M $T/a.png -format 'alpha image, pixel x=0: %[pixel:p{0,0}], x=119: %[pixel:p{119,0}]\n' info:
echo "--- av1gpu on the same file (--lossless, exact): alpha after each operation"
python3 - <<'PY'
import subprocess, sys, os, numpy as np
sys.path.insert(0,'/home/ubuntu/t'); import avt
from PIL import Image
T='/dev/shm/n3'; B=os.environ.get('BIN',os.path.expanduser('~/pr/build/av1gpu'))
src=np.array(Image.open(f'{T}/rgba.png'))
for o in (['--negate'],['--level','10%,90%'],['--gamma','2'],['--auto-level'],['--grayscale','Rec709Luma']):
    subprocess.run([B,f'{T}/rgba.png',f'{T}/q.avif','--lossless']+o,check=True,capture_output=True)
    c=avt.check(f'{T}/q.avif'); print(' '.join(o), 'alpha unchanged' if np.array_equal(c['aplane'],src[...,3]) else 'alpha CHANGED')
PY
echo "=== G5: which filter real magick picks (-define filter:verbose=1)"
for f in rgba.png rgb.png gray.png pal.png c.jpg g.jpg; do for g in 50% 200% 100x100!; do
  r=$($M $T/$f -define filter:verbose=1 -resize $g null: 2>&1 | grep -E '^# (filter|window|support-|B,C|blur|practical)' | tr '\n' ' ' | sed 's/# //g' | cut -c1-130)
  echo "$f -resize $g: $r"; done; done
identify=$($M identify -format '%f class=%r channels=%[channels]\n' $T/rgba.png $T/rgb.png $T/gray.png $T/pal.png $T/c.jpg $T/g.jpg); echo "$identify"
echo "--- the luma gap that makes: --resize 50% vs magick, RGBA / gray / palette / RGB PNG (mean abs. luma, 8-bit)"
python3 - <<'PY'
import subprocess, os, numpy as np, glob
from PIL import Image
T='/dev/shm/n3'; M=os.path.expanduser('~/im/bin/magick'); B=os.environ.get('BIN',os.path.expanduser('~/pr/build/av1gpu')); env=dict(os.environ,LD_LIBRARY_PATH=os.path.expanduser('~/im/lib'))
src=np.array(Image.open(sorted(glob.glob(os.path.expanduser('~/images_std/*.jpg')))[0]).convert('RGB'))[:300,:400]
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng, chunk
al=np.full(src.shape[:2],255,np.uint8); al[:6,:6]=254
wpng(f'{T}/p_rgba.png',np.dstack([src,al])); wpng(f'{T}/p_rgb.png',src); wpng(f'{T}/p_gray.png',src[...,1])
q=Image.fromarray(src).quantize(64); idx=np.array(q,dtype=np.uint8); pal=np.array(q.getpalette()[:192],np.uint8)
wpng(f'{T}/p_pal.png',idx,ct=3,extra=chunk(b'PLTE',pal.tobytes()))
for f in ('p_rgb','p_rgba','p_gray','p_pal'):
    subprocess.run([M,f'{T}/{f}.png','-resize','50%','-depth','8',f'PNG24:{T}/m.png'],check=True,env=env)
    subprocess.run([B,f'{T}/{f}.png',f'{T}/q.avif','--yuv','444','--dump-source',f'{T}/s.yuv','--small-preset','none','--preset','turbo3','--resize','50%'],check=True,capture_output=True)
    m=np.array(Image.open(f'{T}/m.png').convert('RGB')).astype(float); h,w=m.shape[:2]
    yy=np.rint(.299*m[...,0]+.587*m[...,1]+.114*m[...,2])
    y=np.frombuffer(open(f'{T}/s.yuv','rb').read(),np.uint8)[:w*h].reshape(h,w).astype(float)
    print(f, 'mean', round(float(np.abs(y-yy).mean()),3), 'max', float(np.abs(y-yy).max()))
PY
echo N3G5_DONE
