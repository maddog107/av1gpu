#!/bin/bash
export T=/dev/shm/two; source ~/t/try.sh; export LD_LIBRARY_PATH=$HOME/im/lib; IM=~/im/bin/magick
python3 - <<'PY'
from PIL import Image
T='/dev/shm/two'
for name,box in (("lr",(0,0,200,300)),("tb",(0,0,400,150))):
    im=Image.new("RGB",(400,300),(255,255,255)); im.paste((0,0,0),box); im.save(f"{T}/two_{name}.png")
    im.save(f"{T}/two_{name}.jpg",quality=95)
PY
for f in two_lr.png two_tb.png two_lr.jpg two_tb.jpg; do
  $IM $T/$f -format "$f: magick box %@ " info: 2>/dev/null; $IM $T/$f -trim -format "after -trim: %wx%h, opaque pixels %[fx:mean.a]\n" info: 2>&1 | tail -1 | cut -c1-110
  try $B $T/$f $T/o.avif --trim --skip-blank 99
  try $B $T/$f $T/o.avif --trim
  try $B $T/$f $T/o.avif --trim --fuzz 10% --skip-blank 99
done
rm -rf $T/o; ls $T/two_*.png ~/adv/trim/bar_mid_200x120.png > $T/l; $B --batch $T/l --outdir $T/o --trim --skip-blank 99 > /dev/null 2> $T/e; echo "batch: status $? files: $(ls $T/o | tr '\n' ' ')"; grep ': blank' $T/e | cut -c1-110; grep 'trim:' $T/e | cut -c1-100
