#!/bin/bash
# lossless palette check: branch av1gpu --lossless (palette auto / off) on a set; sizes against main's files; every
# output decoded by dav1d and libaom (avifdec) and compared with the source pixels
SET=${1:-screen}; EXTRA=${2:-}
B=~/br/build/av1gpu
O=~/pal/lltest/$SET$EXTRA; rm -rf $O; mkdir -p $O/auto $O/off
ls ~/pal/sets/$SET/*.png > $O/list.txt
$B --batch $O/list.txt --outdir $O/auto --lossless $EXTRA > $O/auto.log 2>&1 || { echo "auto failed"; tail -3 $O/auto.log; }
$B --batch $O/list.txt --outdir $O/off --lossless --palette off $EXTRA > $O/off.log 2>&1 || { echo "off failed"; tail -3 $O/off.log; }
python3 - "$O" "$SET" "$EXTRA" <<'EOF'
import os, subprocess, sys, numpy as np
from PIL import Image
O, st, extra = sys.argv[1], sys.argv[2], sys.argv[3]
H = os.path.expanduser("~")
mode = "ycgco" if "ycgco" in extra else "gbr"
main = f"{H}/pal/out/gpu_ll/{mode}/{st}"
tot = {"auto": 0, "off": 0, "main": 0}
bad = 0
for f in sorted(os.listdir(f"{H}/pal/sets/{st}")):
    if not f.endswith(".png"): continue
    src = np.asarray(Image.open(f"{H}/pal/sets/{st}/{f}").convert("RGB"))
    for k in ("auto", "off"):
        a = f"{O}/{k}/{f}.avif"
        if not os.path.exists(a): print("missing", k, f); bad += 1; continue
        tot[k] += os.path.getsize(a)
        if k == "auto":
            for codec in ("dav1d", "aom"):
                p = f"{O}/{k}/{f}.{codec}.png"
                subprocess.run([f"{H}/opt/bin/avifdec", "-c", codec, "-d", "8", a, p], capture_output=True)
                ok = os.path.exists(p) and np.array_equal(np.asarray(Image.open(p).convert("RGB")), src)
                if not ok: print("NOT EXACT", codec, f); bad += 1
                if os.path.exists(p): os.remove(p)
    m = f"{main}/{f}.avif"
    tot["main"] += os.path.getsize(m) if os.path.exists(m) else 0
print(st, extra, {k: v for k, v in tot.items()}, "auto/main %.3f" % (tot["auto"] / max(1, tot["main"])), "off==main" if tot["off"] == tot["main"] else "off!=main", "bad", bad)
EOF
