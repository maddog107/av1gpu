#!/usr/bin/env python3
"""b08c856: a 4:2:0 JPEG cropped then grayed keeps its chroma upsampling at even offsets; against magick -crop -colorspace Gray"""
import os, subprocess, sys
import numpy as np
from PIL import Image
H=os.path.expanduser('~'); M=f'{H}/im/bin/magick'; env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/cg'; os.makedirs(T,exist_ok=True)
BINS={'pr':f'{H}/pr/build/av1gpu','fix3':f'{H}/fix3/build/av1gpu'}
src=f'{H}/tone2fix3/lineart_q90_420.jpg'
print('crop | op | pr mean (max) | fix3 mean (max)')
for crop in ('400x600+0+0','200x300+10+20','200x300+11+20','200x300+10+21','201x301+10+20','399x599+1+1'):
    for op,iop in ((['--colorspace','Gray'],['-colorspace','Gray']),(['--grayscale','Lightness'],['-grayscale','Lightness']),(['--negate'],['-negate']),(['--level','10%,90%'],['-level','10%,90%'])):
        subprocess.run([M,src,'-crop',crop,'+repage']+iop+['-depth','8',f'PNG24:{T}/m.png'],check=True,env=env)
        m=np.array(Image.open(f'{T}/m.png').convert('RGB')).astype(float); h,w=m.shape[:2]
        yy=np.clip(np.rint(.299*m[...,0]+.587*m[...,1]+.114*m[...,2]),0,255)
        r=[]
        for k,b in BINS.items():
            subprocess.run([b,src,f'{T}/o.avif','--dump-source',f'{T}/d.yuv','--preset','turbo3','--crop',crop]+op,check=True,capture_output=True)
            y=np.frombuffer(open(f'{T}/d.yuv','rb').read(),np.uint8)[:w*h].reshape(h,w).astype(float); e=np.abs(y-yy); r.append(f'{e.mean():.2f} ({e.max():.0f})')
        print(f"{crop:14s} {' '.join(op):24s} {r[0]:12s} {r[1]}")
