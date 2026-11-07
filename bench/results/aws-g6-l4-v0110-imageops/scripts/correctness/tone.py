#!/usr/bin/env python3
"""tone operators and grayscale against (a) the formula, in Python, on exact samples (--lossless, or --dump-source) and (b) magick."""
import glob, json, os, re, subprocess, sys, tempfile, shutil, collections
import concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
TD = tempfile.mkdtemp(prefix="tone_", dir="/dev/shm")
ops = {
 "neg": (["--negate"], ["-negate"]),
 "g2": (["--gamma", "2"], ["-gamma", "2"]),
 "g.7": (["--gamma", "0.7"], ["-gamma", "0.7"]),
 "g.1": (["--gamma", "0.1"], ["-gamma", "0.1"]),
 "g10": (["--gamma", "10"], ["-gamma", "10"]),
 "l1090": (["--level", "10%,90%"], ["-level", "10%,90%"]),
 "l2080.5": (["--level", "20%,80%,0.5"], ["-level", "20%,80%,0.5"]),
 "l5": (["--level", "5%"], ["-level", "5%"]),
 "lq": (["--level", "1000,60000"], ["-level", "1000,60000"]),
 "lx": (["--level", "10%x90%"], ["-level", "10%x90%"]),
 "lg": (["--level", "0%,100%,2.2"], ["-level", "0%,100%,2.2"]),
 "al": (["--auto-level"], ["-auto-level"]),
 "al+neg": (["--auto-level", "--negate"], ["-auto-level", "-negate"]),
 "l+g+n": (["--level", "10%,90%", "--gamma", "1.5", "--negate"], ["-level", "10%,90%", "-gamma", "1.5", "-negate"]),
 "al+l+g": (["--auto-level", "--level", "5%,95%,1.2", "--gamma", "0.8"], ["-auto-level", "-level", "5%,95%,1.2", "-gamma", "0.8"]),
}
gray = {m: (["--grayscale", m], ["-grayscale", m]) for m in ("Rec601Luma", "Rec709Luma", "Average", "Brightness", "Lightness", "RMS")}
gray["csGray"] = (["--colorspace", "Gray"], ["-colorspace", "Gray"])

def formula(rgb, opts, lo=None, hi=None):
    v = rgb.astype(np.float64)
    i = 0
    al = "--auto-level" in opts
    s, o, e = 1.0, 0.0, 1.0
    if al and hi > lo: s = 1 / (hi - lo); o = -lo * s
    if "--level" in opts:
        p = opts[opts.index("--level") + 1]
        a = re.split("[,x]", p)
        def fr(x): return float(x[:-1]) / 100 if x.endswith("%") else float(x) / 65535
        B = fr(a[0]); W_ = fr(a[1]) if len(a) > 1 else 1 - B; G = float(a[2]) if len(a) > 2 else 1
        r = 1 / (W_ - B); s *= r; o = (o - B) * r; e /= G
    if "--gamma" in opts: e /= float(opts[opts.index("--gamma") + 1])
    v = np.clip(v * s + o, 0, 1)
    if e != 1: v = v ** e
    if "--negate" in opts: v = 1 - v
    return v

def mag(src, a, dst, depth=8):
    r = subprocess.run([MAG, src] + a + ["-depth", str(depth), f"png:{dst}"], capture_output=True, text=True, env=env)
    if r.returncode: raise Exception(r.stderr[-200:])
    return np.array(Image.open(dst).convert("RGB"))

res = collections.defaultdict(list)
imgs = [f"{H}/adv/trim/hgrad_256x64.png", f"{H}/adv/trim/noise_257x65.png", f"{H}/adv/trim/corners_1_200x120.png"] + sorted(glob.glob(f"{H}/pr/tests/gate_images/*.png"))[:6]
def one(job):
    img, name, av, im = job
    a = np.array(Image.open(img).convert("RGB"))
    d = tempfile.mkdtemp(dir=TD)
    out = []
    try:
        # lossless: exact samples
        o = f"{d}/o.avif"
        p = subprocess.run([BIN, img, o, "--lossless"] + av, capture_output=True, text=True, timeout=120)
        if p.returncode: return [("ll", name, "rc", p.returncode, p.stderr[-160:])]
        c = avt.check(o, d)
        if name in gray:
            dec = c["planes"][0].astype(int)
            if a.size:
                mp = {"Rec601Luma": lambda x: .298839 * x[..., 0] + .586811 * x[..., 1] + .114350 * x[..., 2], "Rec709Luma": lambda x: .212656 * x[..., 0] + .715158 * x[..., 1] + .072186 * x[..., 2],
                      "csGray": lambda x: .212656 * x[..., 0] + .715158 * x[..., 1] + .072186 * x[..., 2],
                      "Average": lambda x: x.mean(2), "Brightness": lambda x: x.max(2), "Lightness": lambda x: (x.max(2) + x.min(2)) / 2, "RMS": lambda x: np.sqrt((x ** 2).sum(2) / 3)}[name]
                want = mp(a.astype(np.float64))
            diff = np.abs(dec - want)
            out.append(("ll-formula", name, float(diff.mean()), float(diff.max()), os.path.basename(img)))
            m = np.array(Image.open(_m(img, im, d)).convert("RGB"))[..., 0].astype(int)
            df = np.abs(dec - m); out.append(("ll-magick", name, float(df.mean()), int(df.max()), os.path.basename(img)))
        else:
            lo, hi = a.min() / 255, a.max() / 255
            want = formula(a / 255, av, lo, hi) * 255
            dec = np.rint(avt.to_rgb(c["si"], c["planes"]) * 255)
            diff = np.abs(dec - want)
            out.append(("ll-formula", name, float(diff.mean()), float(diff.max()), os.path.basename(img)))
            m = mag(img, im, f"{d}/m.png").astype(int)
            df = np.abs(dec.astype(int) - m); out.append(("ll-magick", name, float(df.mean()), int(df.max()), os.path.basename(img)))
        # 4:4:4 lossy planes against magick (dump)
        ds = f"{d}/s.yuv"
        p = subprocess.run([BIN, img, o, "--yuv", "444", "--dump-source", ds, "--small-preset", "none", "--preset", "turbo3"] + av, capture_output=True, text=True, timeout=120)
        if p.returncode: return out + [("444", name, "rc", p.returncode, p.stderr[-160:])]
        h, w = a.shape[:2]
        if name in gray:
            y = np.frombuffer(open(ds, "rb").read(), np.uint8).reshape(h, w).astype(int)
            m = np.array(Image.open(_m(img, im, d)).convert("RGB"))[..., 0].astype(int)
            df = np.abs(y - m)
        else:
            pl = np.frombuffer(open(ds, "rb").read(), np.uint8).reshape(3, h, w).astype(np.float64)
            m = mag(img, im, f"{d}/m.png").astype(np.float64)
            yy = .299 * m[..., 0] + .587 * m[..., 1] + .114 * m[..., 2]
            df = np.abs(pl[0] - yy)
        out.append(("444-magick-luma", name, float(df.mean()), float(df.max()), os.path.basename(img)))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out

def _m(img, im, d):
    dst = f"{d}/m.png"
    r = subprocess.run([MAG, img] + im + ["-depth", "8", f"png:{dst}"], capture_output=True, text=True, env=env)
    return dst

jobs = [(i, n, a, b) for i in imgs for n, (a, b) in {**ops, **gray}.items()]
agg = collections.defaultdict(list)
with cf.ThreadPoolExecutor(3) as tp:
    for r in tp.map(one, jobs):
        for row in r:
            agg[(row[0], row[1])].append(row[2:])
for k in sorted(agg):
    v = agg[k]
    if any(isinstance(x[0], str) for x in v):
        print(k, "ERR", v[:2]); continue
    print(f"{k[0]:18s} {k[1]:12s} mean {np.mean([x[0] for x in v]):7.3f}  max {max(x[1] for x in v):6.2f}  worst {max(v, key=lambda x: x[1])[2]}")
shutil.rmtree(TD, ignore_errors=True)
