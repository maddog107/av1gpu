import sys, os, subprocess
sys.path.insert(0, os.path.expanduser('~/t'))
from gen_lib import wpng
import numpy as np, avt
from PIL import Image
H=os.path.expanduser('~'); MAG=f'{H}/im/bin/magick'; BIN=os.environ.get('BIN',f'{H}/pr/build/av1gpu')
env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/ar'; os.makedirs(T,exist_ok=True)
rng=np.random.default_rng(4)
h,w=60,80
c=rng.integers(0,256,(h,w,3),dtype=np.uint8)      # random colour under alpha 0
a=np.zeros((h,w),np.uint8)
a[15:45,20:60]=255
c[15:45,20:60]=(250,40,40)
wpng(f'{T}/h.png',np.dstack([c,a]))
g=np.tile(np.linspace(0,255,w).astype(np.uint8)[None,:],(h,1)); wpng(f'{T}/g.png',np.dstack([np.full((h,w,3),(250,40,40),np.uint8),g]))
s=np.zeros((h,w),np.uint8); s[::3]=255; s[:,::4]=255; wpng(f'{T}/s.png',np.dstack([np.full((h,w,3),(30,200,60),np.uint8),s]))
for f in ('h','g','s'):
  for geo in ('300%','150%','237x93!','50%','33%','200%x50%'):
    for mode in (['--lossless'],['--yuv','444','--qindex','8','--alpha-quality','100','--small-preset','none'],['--qindex','8','--alpha-quality','100','--small-preset','none'],['--yuv','444','--depth','10','--qindex','8','--small-preset','none','--alpha-quality','100']):
        o=f'{T}/o.avif'
        p=subprocess.run([BIN,f'{T}/{f}.png',o,'--resize',geo]+mode,capture_output=True,text=True)
        if p.returncode: print(f,geo,mode,'rc',p.returncode,p.stderr[-100:]); continue
        c1=avt.check(o); rgb=avt.to_rgb(c1['si'],c1['planes'])*255; al=c1['aplane'].astype(float)/((1<<c1['si'][3])-1)*255
        want=np.array((250,40,40) if f!='s' else (30,200,60),float)
        vis=al>=8
        e=np.abs(rgb-want).max(2)[vis]
        print(f"{f} {geo:9s} {'/'.join(mode[:3]):28s} visible px {vis.sum():6d}: colour error mean {e.mean():6.2f} max {e.max():6.1f}; alpha min/max {al.min():.0f}/{al.max():.0f}")
