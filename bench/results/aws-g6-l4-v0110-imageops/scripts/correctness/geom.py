#!/usr/bin/env python3
"""--resize's output size against magick -resize, geometry by geometry, on PNGs (batch list with resize= per line, plus the same on the command line for a sample)."""
import os, re, subprocess, sys, struct, tempfile, itertools, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/t"))
from gen_lib import wpng
H = os.path.expanduser("~"); MAG = f"{H}/im/bin/magick"; BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu")
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
def ispe(p):
    d = open(p, "rb").read(); i = d.find(b"ispe"); return struct.unpack(">II", d[i + 8:i + 16])
sizes = [(211, 149), (149, 211), (640, 480), (100, 100), (1000, 37), (37, 1000), (33, 65), (8, 8), (1, 50), (50, 1), (2, 3)]
geos = ["100x", "x100", "100", "100x100", "100x100!", "100x100^", "100x100>", "100x100<", "100x100^>", "100x100^<", "100x100!>", "100x100!<",
        "50%", "150%", "200%", "33.3%", "50%x25%", "25%x50%", "x50%", "x150%", "50%x", "150%x", "100x50%", "50x100%", "50%>", "50%<", "150%>", "150%<", "50%!", "150%!", "50%^", "150%^",
        "@5000", "@100000", "@1000", "@100", "@10", "@1", "@5000>", "@5000<", "@100000>", "@100000<", "@5000!", "@5000^",
        "100!", "x100!", "100^", "x100^", "100>", "100<", "x100>", "x100<", "1", "x1", "1x1", "1x1!", "1x1^", "10x1000", "1000x10", "10x1000^", "1000x10^",
        "10x1000!", "1000x10!", "10x1000>", "1000x10>", "10x1000<", "1000x10<", "211x149", "211x149!", "211x149^", "211x149>", "149x211", "211", "x149", "100.4x", "100.6x", "100.6x50.5!", "x100.6", "100%", "100%x100%"]
wd = tempfile.mkdtemp(prefix="geom_", dir=H)
rows = []; lines = []
for (w, h) in sizes:
    src = f"{wd}/s{w}x{h}.png"
    a = np.random.default_rng(w * 1000 + h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    wpng(src, a)
    for gi, g in enumerate(geos):
        r = subprocess.run([MAG, src, "-resize", g, "-format", "%w %h", "info:"], capture_output=True, text=True, env=env)
        m = re.match(r"(\d+) (\d+)", r.stdout)
        want = (int(m[1]), int(m[2])) if m else None
        link = f"{wd}/g{w}x{h}_{gi:03d}.png"; os.link(src, link)
        lines.append(f"{link}\tresize={g}\n")
        rows.append((w, h, g, want, link, r.stderr.strip()[:80]))
lst = f"{wd}/list"; open(lst, "w").writelines(lines)
out = f"{wd}/out"
r = subprocess.run([BIN, "--batch", lst, "--outdir", out, "--preset", "turbo3", "--max-mp", "100", "--q", "120"] if False else [BIN, "--batch", lst, "--outdir", out, "--preset", "turbo3", "--max-mp", "100"], capture_output=True, text=True)
print("batch rc", r.returncode, r.stderr.strip().splitlines()[-3:] if r.stderr else "")
open(f"{wd}/batch.err", "w").write(r.stderr)
same = 0; diffs = {}
for w, h, g, want, link, err in rows:
    o = f"{out}/{os.path.basename(link)}.avif"
    got = ispe(o) if os.path.exists(o) else None
    if got == want: same += 1
    else: diffs.setdefault(g, []).append(((w, h), want, got))
print(f"{same}/{len(rows)} sizes equal; geometries with a difference: {len(diffs)}")
for g, l in diffs.items():
    print(f"  {g!r}: {len(l)}/{len(sizes)}  e.g. " + "; ".join(f"{s}: IM {w} av1gpu {a}" for s, w, a in l[:3]))
print(wd)
