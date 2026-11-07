import sys, subprocess, re, os
sys.path.insert(0, os.path.expanduser('~/t'))
from gen_lib import wpng
import numpy as np, avt
H=os.path.expanduser('~'); MAG=f'{H}/im/bin/magick'; BIN=f'{H}/pr/build/av1gpu'
env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib')
T='/dev/shm/r4'; os.makedirs(T,exist_ok=True)
a=np.full((240,160,3),235,np.uint8)   # page
a[0,-1]=255                   # top-right corner
a[20:120,30:60]=30            # the picture, in the left part
a[150:152,100:140]=238        # a band right of the picture: differs only from the bottom-right corner (fuzz 10% = 36 grey steps)
a[200:]=205                   # bottom strip in the bottom-left colour, runs to the right edge
a[-1,-1]=200                  # bottom-right corner
p=f'{T}/rule4c.png'; wpng(p,a)
for fz in ('10%',):
    r=subprocess.run([MAG,p,'-fuzz',fz,'-format','%@','info:'],capture_output=True,text=True,env=env).stdout
    subprocess.run([BIN,p,f'{T}/o.avif','--lossless','--trim','--fuzz',fz],check=True,capture_output=True)
    c=avt.check(f'{T}/o.avif')
    print('magick',r,' av1gpu',c['si'][1:3])
