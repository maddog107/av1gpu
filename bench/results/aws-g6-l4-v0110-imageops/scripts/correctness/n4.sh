#!/bin/bash
export T=/dev/shm/n4; source ~/t/try.sh
python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
a=np.full((70000,1000,3),255,np.uint8); a[100:60000,100:900]=(30,120,200); wpng('/dev/shm/n4/tall.png',a)
PY
for o in "" "--yuv 444" "--lossless"; do try $M $T/tall.png $T/o.avif --preset turbo3 --crop 1000x60000+0+0 $o; done
for o in "--negate" "--yuv 444 --negate" "--lossless --negate" "--auto-level" "--yuv 444 --auto-level" "--grayscale Average" "--trim"; do try $B $T/tall.png $T/o.avif --preset turbo3 --crop 1000x60000+0+0 $o; done
