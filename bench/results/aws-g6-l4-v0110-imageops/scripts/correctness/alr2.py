#!/usr/bin/env python3
"""--auto-level at 4:2:0 on JPEG pages: the range each build reads against magick's, and the result"""
import glob, os, subprocess, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser('~/t'))
from trimref_lib import chroma_up
H=os.path.expanduser('~'); M=f'{H}/im/bin/magick'; env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/alr2'; os.makedirs(T,exist_ok=True)
BINS={'pr':f'{H}/pr/build/av1gpu','fix':f'{H}/fix/build/av1gpu','fix3':f'{H}/fix3/build/av1gpu'}
def rng_from(b,w,h,up):
    cw,ch=(w+1)//2,(h+1)//2
    Y=b[:w*h].reshape(h,w).astype(np.int64); U=b[w*h:w*h+cw*ch].reshape(ch,cw).astype(np.int64); V=b[w*h+cw*ch:w*h+2*cw*ch].reshape(ch,cw).astype(np.int64)
    if up: U=chroma_up(U,w,h); V=chroma_up(V,w,h)
    else: U=np.repeat(np.repeat(U,2,0),2,1)[:h,:w]; V=np.repeat(np.repeat(V,2,0),2,1)[:h,:w]
    y=Y/255.; cb=(U-128.)/255.; cr=(V-128.)/255.
    r=np.clip(y+1.402*cr,0,1); g=np.clip(y-.344136286*cb-.714136286*cr,0,1); bb=np.clip(y+1.772*cb,0,1)
    a=np.dstack([r,g,bb])*255; return a.min(), a.max()
files=sorted(glob.glob(f'{H}/comics/*.jpg'))[:12]+[f'{H}/tone2/lineart_q90_420.jpg',f'{H}/tone2/lineart_q90_444.jpg']
print('image | sampling | magick decode range | planes nearest | planes fancy | mean |dY| vs magick: pr / fix')
for f in files:
    samp=subprocess.run([M,'identify','-format','%[jpeg:sampling-factor]',f],capture_output=True,text=True,env=env).stdout.strip()
    a=np.array(Image.open(f).convert('RGB')) if False else None
    raw=subprocess.run([M,f,'-depth','8','rgb:-'],capture_output=True,env=env).stdout
    wh=subprocess.run([M,f,'-format','%w %h','info:'],capture_output=True,text=True,env=env).stdout.split(); w,h=int(wh[0]),int(wh[1])
    a=np.frombuffer(raw,np.uint8).reshape(h,w,3)
    subprocess.run([M,f,'-auto-level','-depth','8',f'PNG24:{T}/m.png'],check=True,env=env)
    m=np.array(Image.open(f'{T}/m.png').convert('RGB')).astype(float); yy=np.rint(.299*m[...,0]+.587*m[...,1]+.114*m[...,2])
    subprocess.run([BINS['fix'],f,f'{T}/o.avif','--dump-source',f'{T}/s.yuv','--preset','turbo3'],check=True,capture_output=True)
    b=np.frombuffer(open(f'{T}/s.yuv','rb').read(),np.uint8)
    rn=rng_from(b,w,h,False); rf=rng_from(b,w,h,True)
    e=[]
    for k in ('pr','fix','fix3'):
        subprocess.run([BINS[k],f,f'{T}/o.avif','--dump-source',f'{T}/s.yuv','--preset','turbo3','--auto-level'],check=True,capture_output=True)
        y=np.frombuffer(open(f'{T}/s.yuv','rb').read(),np.uint8)[:w*h].reshape(h,w).astype(float); e.append(np.abs(y-yy).mean())
    print(f"{os.path.basename(f)[:30]:30s} {samp:12s} magick {a.min()}..{a.max()}  nearest {rn[0]:.1f}..{rn[1]:.1f}  fancy {rf[0]:.1f}..{rf[1]:.1f}  pr {e[0]:.2f} fix {e[1]:.2f} fix3 {e[2]:.2f}", flush=True)
