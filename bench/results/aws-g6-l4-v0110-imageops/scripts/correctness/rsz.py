#!/usr/bin/env python3
"""--resize pixels against magick -resize (planes at 4:4:4 from --dump-source vs luma of magick's output; --lossless: RGB directly), own images."""
import glob, os, subprocess, sys, tempfile, shutil, collections
import concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
TD = tempfile.mkdtemp(prefix="rsz_", dir="/dev/shm")
imgs = sorted(glob.glob(f"{H}/images_std/*.jpg"))[:6] + sorted(glob.glob(f"{H}/comics/*.jpg"))[:3] + sorted(glob.glob(f"{H}/pr/tests/gate_images/*.png"))[:4] + sorted(glob.glob(f"{H}/adv/trim/noise_257x65.png"))
alpha = sorted(glob.glob(f"{H}/adv/alpha/agrad_120x80.png")) + sorted(glob.glob(f"{H}/adv/alpha/hidden_120x80.png"))
geos = ["150%", "50%", "300x300^", "100x100!", "200%", "33%", "37x", "x41", "300x20!", "20x300!", "@5000", "173%"]
def one(j):
    img, g = j
    d = tempfile.mkdtemp(dir=TD); out = []
    try:
        im = Image.open(img)
        m = f"{d}/m.png"
        r = subprocess.run([MAG, img, "-resize", g, "-depth", "8", f"png:{m}"], capture_output=True, text=True, env=env)
        if r.returncode: return [("skip", g, r.stderr[:80])]
        ma = np.array(Image.open(m).convert("RGB")).astype(np.float64)
        mh, mw = ma.shape[:2]
        yy = .299 * ma[..., 0] + .587 * ma[..., 1] + .114 * ma[..., 2]
        o, ds = f"{d}/o.avif", f"{d}/s.yuv"
        p = subprocess.run([BIN, img, o, "--yuv", "444", "--dump-source", ds, "--small-preset", "none", "--preset", "turbo3", "--resize", g, "--max-mp", "200"], capture_output=True, text=True, timeout=300)
        if p.returncode: return [("rc", g, p.stderr[-100:])]
        c = avt.parse(o); si = avt.seq_info(c["colour"]); w, h = si[1], si[2]
        if (w, h) != (mw, mh): return [("size", g, (w, h, mw, mh))]
        b = np.frombuffer(open(ds, "rb").read(), np.uint8)
        pl = b[:w * h].reshape(h, w).astype(np.float64)
        df = np.abs(pl - yy)
        kind = "png" if img.endswith(".png") else "jpeg"
        out.append((kind, g, float(df.mean()), float(df.max()), os.path.basename(img)))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out
agg = collections.defaultdict(list)
with cf.ThreadPoolExecutor(3) as tp:
    for r in tp.map(one, [(i, g) for i in imgs for g in geos]):
        for row in r: agg[(row[0], row[1])].append(row[2:])
for k in sorted(agg, key=str):
    v = agg[k]
    if k[0] in ("skip", "rc", "size"): print(k, v[:2]); continue
    print(f"{k[0]:6s} {k[1]:10s} n={len(v):3d} mean |dY| {np.mean([x[0] for x in v]):6.3f}  worst-mean {max(x[0] for x in v):6.3f} max {max(x[1] for x in v):6.1f}  {max(v, key=lambda x: x[1])[2]}")
shutil.rmtree(TD, ignore_errors=True)
