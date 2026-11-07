import subprocess, os, sys, numpy as np
from PIL import Image
sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import avt
T='/dev/shm/n3'; os.makedirs(T,exist_ok=True); M=os.path.expanduser('~/im/bin/magick'); B=os.environ.get('BIN',os.path.expanduser('~/pr/build/av1gpu')); env=dict(os.environ,LD_LIBRARY_PATH=os.path.expanduser('~/im/lib'))
h,w=80,120
c=np.random.default_rng(2).integers(60,181,(h,w,3),dtype=np.uint8)
print('RGB range of the source', c.min(), c.max())
for tag,al in (('alpha 0..255',np.tile(np.linspace(0,255,w).astype(np.uint8)[None,:],(h,1))),('alpha 100..200',np.tile(np.linspace(100,200,w).astype(np.uint8)[None,:],(h,1))),('all alpha 254',np.full((h,w),254,np.uint8))):
    wpng(f'{T}/al.png',np.dstack([c,al]))
    for ops in (['-auto-level'],['-channel','RGB','-auto-level','+channel']):
        subprocess.run([M,f'{T}/al.png']+ops+[f'PNG32:{T}/o.png'],check=True,env=env)
        o=np.array(Image.open(f'{T}/o.png').convert('RGBA'))
        print(f"{tag:15s} {' '.join(ops):34s} output RGB range {o[...,:3].min()}..{o[...,:3].max()}, alpha range {o[...,3].min()}..{o[...,3].max()}")
    subprocess.run([B,f'{T}/al.png',f'{T}/q.avif','--lossless','--auto-level'],check=True,capture_output=True)
    q=avt.check(f'{T}/q.avif'); r=np.rint(avt.to_rgb(q['si'],q['planes'])*255)
    print(f"{tag:15s} av1gpu --auto-level                 output RGB range {int(r.min())}..{int(r.max())}, alpha range {q['aplane'].min()}..{q['aplane'].max()}")
