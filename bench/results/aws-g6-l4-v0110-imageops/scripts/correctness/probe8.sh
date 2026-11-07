#!/bin/bash
export T=/dev/shm/p8; source ~/t/try.sh
echo "== F1 in batch: --resize 50% with EXIF 6; plain and resize= line"
rm -rf $T/o; printf '%s\n' ~/adv/jpeg/ex6_420_160x96.jpg ~/adv/jpeg/ex1_420_160x96.jpg > $T/l; $B --batch $T/l --outdir $T/o --resize 50% 2>&1 | grep -v 'stage ms' | tail -1 | cut -c1-120
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in sorted(glob.glob('$T/o/*.avif')):
    c=avt.check(f); print(f.split('/')[-1], c['si'][:4], 'irot',c['irot'])
PY
echo "== JPEG sides above 6000 (CPU route) with the options"
for f in ~/adv/tall/tj_8x16000.jpg ~/adv/tall/tj_16000x8.jpg ~/adv/tall/tj_64x16000.jpg ~/adv/tall/tj_16000x64.jpg; do for o in "--trim --fuzz 5%" "--negate" "--grayscale Average" "--auto-level --trim --fuzz 5% --resize 25%"; do try $B $f $T/o.avif --preset turbo3 $o; done; done
rm -rf $T/o; ls ~/adv/tall/tj_*.jpg > $T/l; $B --batch $T/l --outdir $T/o --preset turbo3 --trim --fuzz 5% --skip-blank 99 --auto-level 2>&1 | grep -v 'stage ms' | tail -2 | cut -c1-200; ls $T/o | wc -l
echo "== alpha shrink filter: magick Mitchell, av1gpu Lanczos (opaque pixels only)"
python3 - <<'PY'
import sys,subprocess,os; sys.path.insert(0,'/home/ubuntu/t'); import avt
import numpy as np
from PIL import Image
from gen_lib import wpng
H='/home/ubuntu'; env=dict(os.environ,LD_LIBRARY_PATH=f'{H}/im/lib')
src=np.array(Image.open(sorted(__import__('glob').glob(f'{H}/images_std/*.jpg'))[0]).convert('RGB'))[:300,:400]
al=np.full(src.shape[:2],255,np.uint8); al[:6,:6]=254      # one nearly opaque corner: the image has alpha
wpng('/dev/shm/p8/a.png',np.dstack([src,al])); wpng('/dev/shm/p8/o.png',src)
for tag,f in (('alpha','/dev/shm/p8/a.png'),('opaque','/dev/shm/p8/o.png')):
    subprocess.run([f'{H}/im/bin/magick',f,'-resize','50%','-depth','8','png:/dev/shm/p8/m.png'],env=env,check=True)
    subprocess.run([f'{H}/pr/build/av1gpu',f,'/dev/shm/p8/q.avif','--yuv','444','--dump-source','/dev/shm/p8/s.yuv','--small-preset','none','--preset','turbo3','--resize','50%'],check=True,capture_output=True)
    m=np.array(Image.open('/dev/shm/p8/m.png').convert('RGB')).astype(float); h,w=m.shape[:2]
    yy=.299*m[...,0]+.587*m[...,1]+.114*m[...,2]
    y=np.frombuffer(open('/dev/shm/p8/s.yuv','rb').read(),np.uint8)[:w*h].reshape(h,w).astype(float)
    print(tag,'|dY| mean',round(float(np.abs(y-yy).mean()),3),'max',round(float(np.abs(y-yy).max()),1))
PY
