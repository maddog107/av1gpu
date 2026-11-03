#!/usr/bin/env python3
"""Decoder cost of LR: dav1d 1.4.1 (--threads 1) decode time of the q60 files (av1gpu's default operating point) of
s6 (no LR), p6keep and p6full, size range and standard set, 3 passes in rotating order, best of 3 per file; run on
an otherwise idle machine, 32 files at a time.  -> ~/lr/rows/dectime.json"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/bench")
import struct  # noqa: E402

import avif2ivf  # noqa: E402

CFGS = ["s6", "p6keep", "p6full"]
SETS = ["gamut", "std"]
tmp = tempfile.mkdtemp(dir=f"{H}/lr/tmp")


def ivf(p):
    b = open(p, "rb").read()
    obus = avif2ivf.primary_item(b)
    w, h = avif2ivf.ispe(b)
    out = os.path.join(tmp, p.replace("/", "_") + ".ivf")
    hdr = b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", w & 0xFFFF, h & 0xFFFF, 30, 1, 1, 0)
    open(out, "wb").write(hdr + struct.pack("<IQ", len(obus), 0) + obus)
    return out


files = {}
for c in CFGS:
    for s in SETS:
        d = f"{H}/lr/out/420/{c}/{s}"
        for f in sorted(os.listdir(d)):
            if f.endswith(".q60.avif"):
                files[(c, s, f)] = ivf(os.path.join(d, f))


def dec(path):
    t0 = time.perf_counter()
    subprocess.run(["dav1d", "--threads", "1", "-q", "-i", path, "-o", "/dev/null", "--muxer", "null"], check=True)
    return time.perf_counter() - t0


best = {}
for rnd in range(3):
    order = CFGS[rnd:] + CFGS[:rnd]
    for c in order:
        keys = [k for k in files if k[0] == c]
        with cf.ThreadPoolExecutor(32) as ex:
            for k, t in zip(keys, ex.map(lambda k: dec(files[k]), keys)):
                best[k] = min(best.get(k, 1e9), t)
res = {}
for c in CFGS:
    for s in SETS:
        res[f"{c}|{s}"] = sum(t for k, t in best.items() if k[0] == c and k[1] == s)
for c in CFGS:
    for s in SETS:
        print(c, s, f"{res[f'{c}|{s}']:.2f} s", f"{res[f'{c}|{s}'] / res[f's6|{s}']:.3f}x")
json.dump(res, open(f"{H}/lr/rows/dectime.json", "w"), indent=1)
