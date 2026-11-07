#!/bin/bash
export T=/dev/shm/p4; source ~/t/try.sh
P=~/adv/trim/bar_mid_200x120.png; J=~/adv/jpeg/ex0_420_160x96.jpg; A1=~/adv/alpha/hidden_120x80.png
echo "== last of --resize / --max-size"
try $B $P $T/o.avif --max-size 50x50 --resize 100x
try $B $P $T/o.avif --resize 100x --max-size 50x50
try $B $P $T/o.avif --resize 100x --max-height 16
try $B $P $T/o.avif --max-height 16 --resize 100x
echo "== multi-q with in-place ops"
rm -rf $T/mq; printf "%s\n" $J $P > $T/l; $B --batch $T/l --outdir $T/mq --qindex 60,120 --negate --trim --fuzz 5% 2>&1 | grep -v 'stage ms' | tail -3 | cut -c1-160; ls -la $T/mq | cut -c20-
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in sorted(glob.glob('$T/mq/*.avif')):
    c=avt.check(f); y=c['planes'][0]; print(f.split('/')[-1], c['si'][:4], int(y.min()), int(y.max()), round(float(y.mean()),1))
PY
echo "== --check / alternates"
for o in "--check" "--cpu-entropy" "--dev-entropy" "--check-entropy" "--debug-outputs"; do try $B $J $T/o.avif $o --negate --trim --fuzz 5% --resize 150%; done
echo "== max-mp vs trim: batch and single"
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from PIL import Image; import numpy as np
a=np.full((7000,7000,3),255,np.uint8); a[3000:3500,3000:3500]=(20,80,200)
Image.fromarray(a).save('/dev/shm/p4/big49mp.jpg',quality=90)
PY
try $B $T/big49mp.jpg $T/o.avif --trim --max-input-mp 60
try $B $T/big49mp.jpg $T/o.avif --trim --fuzz 5% --max-input-mp 60
rm -rf $T/bm; echo $T/big49mp.jpg > $T/l; $B --batch $T/l --outdir $T/bm --trim --fuzz 5% --max-input-mp 60 2>&1 | grep -v 'stage ms' | tail -3 | cut -c1-200; ls $T/bm 2>/dev/null
echo "== ycgco == gbr box"
for f in ~/adv/trim/bar_mid_200x120.png ~/adv/trim/ring_gray37.png ~/adv/trim/corners_1_200x120.png ~/adv/alpha/hidden_120x80.png ~/adv/alpha/agrad_120x80.png ~/adv/alpha/astep_120x80.png; do for z in 0% 10%; do
  try $B $f $T/o.avif --lossless --trim --fuzz $z; try $B $f $T/o.avif --lossless --ycgco-re --trim --fuzz $z; done; done
echo "== gray with ICC and avifdec"
python3 - <<'PY'
from PIL import Image, ImageCms
import numpy as np
a=np.random.default_rng(1).integers(0,255,(64,64,3),dtype=np.uint8)
icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
Image.fromarray(a).save('/dev/shm/p4/icc.jpg',quality=95,icc_profile=icc)
PY
try $B $T/icc.jpg $T/o.avif --grayscale Rec709Luma; avifdec $T/o.avif $T/o.png 2>&1 | tail -3 | cut -c1-160; avifdec --info $T/o.avif 2>&1 | sed -n 1,14p | cut -c1-120
echo "== alpha-less, gray + alpha"
try $B $A1 $T/o.avif --grayscale Average; try $B $A1 $T/o.avif --grayscale Average --lossless; try $B $A1 $T/o.avif --grayscale Average --lossless --ycgco-re; try $B $A1 $T/o.avif --colorspace Gray --depth 10 --resize 150%
