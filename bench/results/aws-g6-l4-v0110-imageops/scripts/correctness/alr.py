#!/usr/bin/env python3
"""why --auto-level differs at 4:2:0: the range read from the planes against the true RGB range and magick's"""
import glob, os, subprocess, sys
import numpy as np
from PIL import Image
H=os.path.expanduser('~'); B=os.environ.get('BIN',f'{H}/pr/build/av1gpu'); M=f'{H}/im/bin/magick'; env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/alr'; os.makedirs(T,exist_ok=True)
def rgb_from(pl,w,h,f420):
    Y,U,V=pl
    if f420:
        U=np.repeat(np.repeat(U,2,0),2,1)[:h,:w]; V=np.repeat(np.repeat(V,2,0),2,1)[:h,:w]
    y=Y/255.; cb=(U-128.)/255.; cr=(V-128.)/255.
    r=np.clip(y+1.402*cr,0,1); g=np.clip(y-.344136286*cb-.714136286*cr,0,1); b=np.clip(y+1.772*cb,0,1)
    return np.dstack([r,g,b])
files=sorted(glob.glob(f'{H}/tone2/page_*.png'))+[f'{H}/tone2/lineart.png',f'{H}/pr/tests/gate_images/cfl_edge_211x149.png']
print('image | true RGB range | magick -auto-level output range | planes 4:4:4 range | planes 4:2:0 range | av1gpu Y range 4:2:0 / magick luma range')
for f in files:
    a=np.array(Image.open(f).convert('RGB')); h,w=a.shape[:2]
    subprocess.run([M,f,'-auto-level','-depth','8',f'PNG24:{T}/m.png'],check=True,env=env)
    m=np.array(Image.open(f'{T}/m.png').convert('RGB')).astype(float)
    res=[]
    for fm,opts in (('444',['--yuv','444']),('420',[])):
        subprocess.run([B,f,f'{T}/o.avif','--dump-source',f'{T}/s.yuv','--small-preset','none','--preset','turbo3']+opts,check=True,capture_output=True)
        b=np.frombuffer(open(f'{T}/s.yuv','rb').read(),np.uint8)
        if fm=='444': pl=[b[i*w*h:(i+1)*w*h].reshape(h,w).astype(float) for i in range(3)]
        else:
            cw,ch=(w+1)//2,(h+1)//2; pl=[b[:w*h].reshape(h,w).astype(float),b[w*h:w*h+cw*ch].reshape(ch,cw).astype(float),b[w*h+cw*ch:w*h+2*cw*ch].reshape(ch,cw).astype(float)]
        r=rgb_from(pl,w,h,fm=='420'); res.append((r.min()*255,r.max()*255))
    subprocess.run([B,f,f'{T}/o.avif','--dump-source',f'{T}/s.yuv','--small-preset','none','--preset','turbo3','--auto-level'],check=True,capture_output=True)
    y=np.frombuffer(open(f'{T}/s.yuv','rb').read(),np.uint8)[:w*h].reshape(h,w).astype(float)
    yy=np.rint(.299*m[...,0]+.587*m[...,1]+.114*m[...,2])
    print(f"{os.path.basename(f)[:34]:34s} true {a.min()}..{a.max()}  magick out {m.min():.0f}..{m.max():.0f}  444 {res[0][0]:.1f}..{res[0][1]:.1f}  420 {res[1][0]:.1f}..{res[1][1]:.1f}  mean |dY| {np.abs(y-yy).mean():.2f}")
