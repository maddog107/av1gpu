#!/usr/bin/env python3
"""--resize / --max-size with every EXIF orientation: the displayed size against magick -auto-orient -resize (batch and single), and that the
decoded + irot/imir picture matches the upright picture."""
import os, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); MAG = f"{H}/im/bin/magick"; BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu")
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
geos = ["80x", "x80", "80x50", "80x50!", "80x50^", "50%", "150%", "50%x25%", "x50%", "@2000", "80x50>", "40x60>", "200x200<", "100%"]
td = tempfile.mkdtemp(prefix="exg_", dir=H)
bad = 0; n = 0; rows = {}
for w, h in [(160, 96)]:
    for ss in ("420", "444"):
        for o in range(1, 9):
            f = f"{H}/adv/jpeg/ex{o}_{ss}_{w}x{h}.jpg"
            for g in geos:
                r = subprocess.run([MAG, f, "-auto-orient", "-resize", g, "-format", "%w %h", "info:"], capture_output=True, text=True, env=env)
                want = tuple(int(v) for v in r.stdout.split()[:2])
                out = f"{td}/o.avif"
                if os.path.exists(out): os.remove(out)
                p = subprocess.run([BIN, f, out, "--resize", g], capture_output=True, text=True)
                n += 1
                if p.returncode:
                    got = f"rc {p.returncode} {p.stderr[-100:]}"
                else:
                    c = avt.check(out)
                    sw, sh = c["si"][1], c["si"][2]
                    got = (sh, sw) if c["irot"] in (1, 3) else (sw, sh)
                if got != want:
                    bad += 1
                    rows.setdefault(g, []).append((ss, o, want, got))
print(f"{n - bad}/{n} displayed sizes equal magick -auto-orient -resize")
for g, l in rows.items():
    print(f"  {g!r}: {len(l)} differ, e.g. " + "; ".join(f"{ss} o{o}: IM {w} av1gpu {a}" for ss, o, w, a in l[:3]) + "; orientations " + str(sorted({o for _, o, _, _ in l})))
