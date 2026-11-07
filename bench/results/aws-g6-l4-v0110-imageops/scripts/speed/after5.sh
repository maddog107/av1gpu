#!/bin/bash
L=~/results/after.log
until grep -q AFTER4_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) proto builds" >> $L
for p in p1 p2 p3 p4 p5 p6 pc; do bash ~/mkproto.sh $p; done > ~/results/proto_build.txt 2>&1
# a page that is all border: every pixel is read, the worst case for any early exit
python3 - > ~/results/blank.txt 2>&1 <<'PY'
import os
from PIL import Image
d = os.path.expanduser("~/sets/blank"); os.makedirs(d, exist_ok=True)
for i in range(12):
    Image.new("RGB", (2886, 3950), (248 - i, 246, 240)).save(f"{d}/b{i}.jpg", quality=92)
print("blank pages written")
PY
echo "$(date -u +%H:%M:%S) pk" >> $L; python3 ~/pk.py 3 > ~/results/pk.txt 2>&1
echo "$(date -u +%H:%M:%S) AFTER5_DONE" >> $L
