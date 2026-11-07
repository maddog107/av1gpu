#!/bin/bash
export T=/dev/shm/p6; source ~/t/try.sh
P=~/adv/trim/bar_mid_200x120.png; J=~/adv/jpeg/ex0_420_160x96.jpg
for g in 65534x1! 65535x1! 65536x1! 65537x1! 65540x1! 70000x1! 100000x1! 131072x1! 1x65535! 1x65536! 65535x16! 65536x16! 65537x16! ; do try $B $P $T/o.avif --resize $g; avifdec --info $T/o.avif 2>&1 | grep -E 'Resolution|ERROR|rror' | head -2 | cut -c1-120; done
try $B $J $T/o.avif --resize 65537x1!
for s in 65536 65537 70000; do python3 - <<PY
import sys; sys.path.insert(0,'/home/ubuntu/t')
from gen_lib import wpng
import numpy as np
a=np.full((2,$s,3),255,np.uint8); a[:,100:200]=(10,20,200); wpng('$T/w$s.png',a)
PY
try $M $T/w$s.png $T/o.avif; try $B $T/w$s.png $T/o.avif; done
