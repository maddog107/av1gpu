import sys, os, subprocess, glob
sys.path.insert(0, os.path.expanduser('~/t'))
from gen_lib import wpng
import numpy as np, avt
from PIL import Image
H=os.path.expanduser('~'); MAG=f'{H}/im/bin/magick'; BIN=os.environ.get('BIN',f'{H}/pr/build/av1gpu')
env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib'); T='/dev/shm/dp'; os.makedirs(T,exist_ok=True)
rng=np.random.default_rng(3)
v=rng.integers(0,1024,(80,120,3)).astype(np.float32)
s=(0.5+v/1023.0*65535.0).astype(np.float32).astype(np.uint16)
wpng(f'{T}/ten.png',s,bits=16)
a=v.astype(float)/1023
ops={'neg':(['--negate'],lambda v:1-v,['-negate']),'g':(['--gamma','1.7'],lambda v:v**(1/1.7),['-gamma','1.7']),'l':(['--level','10%,90%,1.3'],lambda v:np.clip((v-.1)/.8,0,1)**(1/1.3),['-level','10%,90%,1.3'])}
def im16(f,im):
    raw=subprocess.run([MAG,f]+im+['-depth','16','rgb:-'],capture_output=True,env=env).stdout
    return np.frombuffer(raw,'<u2').reshape(80,120,3).astype(float)/65535
for k,(o,fn,im) in ops.items():
    p=subprocess.run([BIN,f'{T}/ten.png',f'{T}/o.avif','--lossless']+o,capture_output=True,text=True)
    if p.returncode: print(k,'rc',p.returncode,p.stderr[-150:]); continue
    c=avt.check(f'{T}/o.avif'); rgb=avt.to_rgb(c['si'],c['planes'])
    e=np.abs(rgb-fn(a))*1023; e2=np.abs(rgb-im16(f'{T}/ten.png',im))*1023
    print(f'lossless 10-bit {k}: {c["si"][:5]} vs formula mean {e.mean():.3f} max {e.max():.2f}; vs magick (16 bit) mean {e2.mean():.3f} max {e2.max():.2f} (10-bit steps)')
for name in ('rgb16_120x80',):
    f=f'{H}/adv/deep/{name}.png'
    for k,(o,fn,im) in ops.items():
        ds=f'{T}/s.yuv'
        p=subprocess.run([BIN,f,f'{T}/o.avif','--yuv','444','--depth','10','--dump-source',ds,'--small-preset','none','--preset','turbo3']+o,capture_output=True,text=True)
        m=im16(f,im); yy=(.299*m[...,0]+.587*m[...,1]+.114*m[...,2])*1023
        b=np.frombuffer(open(ds,'rb').read(),'<u2'); y=b[:80*120].reshape(80,120).astype(float)
        e=np.abs(y-yy); print(f'{name} depth10 {k} rc {p.returncode}: dump {len(b)} |dY| mean {e.mean():.3f} max {e.max():.2f} (10-bit steps)')
