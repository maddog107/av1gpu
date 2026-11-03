#!/usr/bin/env python3
"""Butteraugli (libjxl 0.11.1 butteraugli_main, 3-norm, as the palette study) of the 4:2:0 files of s6, p6keep, p6full,
s4 and s4r1 (dav1d decode against the source), as a second perceptual metric.  -> ~/lr/rows/butter.jsonl"""
import concurrent.futures as cf
import json
import os
import subprocess
import tempfile

H = os.path.expanduser("~")
LR = f"{H}/lr"
SETS = {"std": f"{H}/img/std", "gamut": f"{H}/img/gamut", "kodak": f"{H}/img/kodak"}
CFGS = ["s6", "p6keep", "p6full", "s4", "s4r1"]
QS = [30, 45, 60, 75, 90]
jobs = []
for c in CFGS:
    for s, d in SETS.items():
        for f in sorted(os.listdir(d)):
            for q in QS:
                jobs.append((c, s, os.path.join(d, f), q, f"{LR}/out/420/{c}/{s}/{f}.q{q}.avif"))
jobs.sort(key=lambda j: -os.path.getsize(j[2]))
_mp = {}


def mp(path):
    if path not in _mp:
        from PIL import Image
        with Image.open(path) as im:
            _mp[path] = im.size[0] * im.size[1] / 1e6
    return _mp[path]


def one(j):
    c, s, src, q, p = j
    r = dict(fmt="420", cfg=c, set=s, image=os.path.basename(src), q=q, bytes=os.path.getsize(p), butteraugli=None)
    with tempfile.TemporaryDirectory(dir=f"{LR}/tmp") as td:
        png = os.path.join(td, "d.png")
        if subprocess.run([f"{H}/opt/bin/avifdec", "-c", "dav1d", p, png], capture_output=True).returncode == 0:
            o = subprocess.run([f"{H}/opt/bin/butteraugli_main", src, png], capture_output=True, text=True)
            for line in o.stdout.splitlines():
                if line.startswith("3-norm"):
                    r["butteraugli"] = float(line.split()[-1])
    return r


# butteraugli_main needs about 170 bytes per pixel (4.2 GB at 24 MP): 64 at once on the largest photos ran the
# 123 GB instance out of memory, so sources of 5 MP and more run 16 at a time, the rest 64 at a time
big = [j for j in jobs if os.path.getsize(j[2]) > 0 and mp(j[2]) >= 5]
small = [j for j in jobs if j not in big]
with open(f"{LR}/rows/butter.jsonl", "w") as f:
    for part, n in ((big, 16), (small, os.cpu_count())):
        with cf.ThreadPoolExecutor(n) as ex:
            for r in ex.map(one, part):
                f.write(json.dumps(r) + "\n")
                f.flush()
print("BUTTER_DONE")
