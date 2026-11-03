#!/bin/bash
# Prove that each libaom config switch takes effect: encode a synthetic screenshot and a Kodak photo with
# configs A / B / C / D / C0 at q60 (and lossless), print md5, size and pstat's frame flags and block counts.
E=~/opt/bin/avifenc
D=~/switch; mkdir -p $D
python3 - <<'EOF'
import os
from PIL import Image, ImageDraw, ImageFont
im = Image.new("RGB", (1280, 720), (255, 255, 255))
d = ImageDraw.Draw(im)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
for k in range(30):
    d.text((20, 10 + 23 * k), f"line {k}: palette mode codes up to eight colours per block plus a colour index map", font=f,
           fill=(20, 20, 20) if k % 5 else (200, 30, 30))
d.rectangle([900, 100, 1200, 400], fill=(40, 90, 200))
im.save(os.path.expanduser("~/switch/screen.png"))
EOF
[ -s $D/kodim05.png ] || curl -s -o $D/kodim05.png https://r0k.us/graphics/kodak/kodak/kodim05.png
declare -A CFG=(
  [A]="-a enable-palette=0 -a enable-intrabc=0"
  [B]=""
  [C]="-a tune-content=screen -a enable-intrabc=0"
  [D]="-a tune-content=screen"
  [C0]="-a tune-content=screen -a enable-palette=0 -a enable-intrabc=0"
)
for img in screen kodim05; do
  for c in A B C D C0; do
    o=$D/$img.$c.avif
    $E -q 60 -y 420 -j 1 --ignore-exif --ignore-xmp -s 6 -a tune=iq ${CFG[$c]} $D/$img.png $o > $D/$img.$c.log 2>&1 || echo "FAIL $img $c"
    python3 ~/scripts/avif2ivf.py $o $o.ivf
    echo "$img $c $(md5sum < $o | cut -c1-12) $(stat -c %s $o) $(~/opt/bin/pstat $o.ivf)"
    o=$D/$img.$c.ll.avif
    $E -l -j 1 -s 6 ${CFG[$c]} $D/$img.png $o > $D/$img.$c.ll.log 2>&1 || echo "FAIL ll $img $c"
    python3 ~/scripts/avif2ivf.py $o $o.ivf
    echo "$img $c-ll $(md5sum < $o | cut -c1-12) $(stat -c %s $o) $(~/opt/bin/pstat $o.ivf)"
  done
done
