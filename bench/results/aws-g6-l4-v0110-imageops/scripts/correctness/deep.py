import sys, os, subprocess, glob
sys.path.insert(0, os.path.expanduser('~/t'))
from gen_lib import wpng
import numpy as np, avt
from PIL import Image
H=os.path.expanduser('~'); MAG=f'{H}/im/bin/magick'; BIN=f'{H}/pr/build/av1gpu'
env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/dp'; os.makedirs(T,exist_ok=True)
def read16(p):
    subprocess.run([MAG,p,'-depth','16',f'rgb:{p}.rgb'],check=True,env=env)
    return None
f=f'{H}/adv/deep/ten16_120x80.png'
# exact: 10-bit lossless, tone in 10-bit GBR
raw=subprocess.run([MAG,f,'-depth','16','rgb:-'],capture_output=True,env=env).stdout
a=np.frombuffer(raw,'>u2').reshape(80,120,3).astype(float)/65535
ops={'neg':(['--negate'],lambda v:1-v,['-negate']),'g':(['--gamma','1.7'],lambda v:v**(1/1.7),['-gamma','1.7']),'l':(['--level','10%,90%,1.3'],lambda v:np.clip((v-.1)/.8,0,1)**(1/1.3),['-level','10%,90%,1.3'])}
for k,(o,fn,im) in ops.items():
    p=subprocess.run([BIN,f,f'{T}/o.avif','--lossless']+o,capture_output=True,text=True)
    if p.returncode: print(k,'rc',p.returncode,p.stderr[-150:]); continue
    c=avt.check(f'{T}/o.avif'); rgb=avt.to_rgb(c['si'],c['planes'])
    e=np.abs(rgb-fn(a))*1023
    raw=subprocess.run([MAG,f]+im+['-depth','16','rgb:-'],capture_output=True,env=env).stdout
    m=np.frombuffer(raw,'>u2').reshape(80,120,3).astype(float)/65535
    e2=np.abs(rgb-m)*1023
    print(f'lossless 10-bit {k}: {c["si"][:5]} vs formula mean {e.mean():.3f} max {e.max():.2f} (10-bit steps); vs magick -depth 16 mean {e2.mean():.3f} max {e2.max():.2f}')
# 16-bit PNG at --depth 10 lossy, 4:4:4 planes vs magick
for name in ('rgb16_120x80','g16_120x80'):
    f=f'{H}/adv/deep/{name}.png'
    for k,(o,fn,im) in ops.items():
        ds=f'{T}/s.yuv'
        p=subprocess.run([BIN,f,f'{T}/o.avif','--yuv','444','--depth','10','--dump-source',ds,'--small-preset','none','--preset','turbo3']+o,capture_output=True,text=True)
        raw=subprocess.run([MAG,f]+im+['-depth','16','-colorspace','sRGB','-type','TrueColor','rgb:-'],capture_output=True,env=env).stdout
        m=np.frombuffer(raw,'>u2')
        if m.size!=80*120*3: print(name,k,'size',m.size); continue
        m=m.reshape(80,120,3).astype(float)/65535
        yy=(.299*m[...,0]+.587*m[...,1]+.114*m[...,2])*1023
        b=np.frombuffer(open(ds,'rb').read(),'<u2'); y=b[:80*120].reshape(80,120).astype(float)
        e=np.abs(y-yy); print(f'{name} depth10 {k}: |dY| mean {e.mean():.3f} max {e.max():.2f} (10-bit steps)')
