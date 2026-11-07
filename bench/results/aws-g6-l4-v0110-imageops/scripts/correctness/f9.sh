#!/bin/bash
export T=/dev/shm/f9; export BIN=$HOME/fix/build/av1gpu; source ~/t/try.sh
python3 - <<'PY'
from PIL import Image
import numpy as np
a=np.full((7000,7000,3),255,np.uint8); a[3000:3500,3000:3500]=(20,80,200)
Image.fromarray(a).save('/dev/shm/f9/big49mp.jpg',quality=90)
PY
for b in $HOME/pr/build/av1gpu $BIN; do
  try $b $T/big49mp.jpg $T/o.avif --trim --fuzz 5% --max-input-mp 60
  rm -rf $T/bm; echo $T/big49mp.jpg > $T/l; $b --batch $T/l --outdir $T/bm --trim --fuzz 5% --max-input-mp 60 > /dev/null 2> $T/e; echo "$(basename $(dirname $(dirname $b))) batch: status $? files $(ls $T/bm 2>/dev/null | wc -l): $(grep skipped $T/e | cut -c1-120)"
done
python3 - <<PY
import sys,glob; sys.path.insert(0,'/home/ubuntu/t'); import avt
for f in glob.glob('$T/bm/*.avif'): c=avt.check(f); print('fix batch output', c['si'][:4])
PY
