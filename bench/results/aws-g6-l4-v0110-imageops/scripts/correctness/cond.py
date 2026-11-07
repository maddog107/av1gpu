#!/usr/bin/env python3
"""the trim conditions decided as the docs say?  An independent model on exact RGB (--lossless PNGs), random parameters.  Plus windows of auto-level, alpha untouched."""
import glob, os, random, re, subprocess, sys, tempfile, shutil, collections, math
import concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
from gen_lib import wpng
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
TD = tempfile.mkdtemp(prefix="cond_", dir="/dev/shm")

def box(a, f):
    """documented box on 8-bit RGB (no alpha): (left, top, right, bottom), None sides when nothing differs"""
    p = a.astype(np.float64) / 255; h, w = a.shape[:2]; thr = 6 * f * f
    d = [((p - p[y, x]) ** 2).sum(2) > thr for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]]
    ys, xs = np.nonzero(d[0]); l = xs.min() if len(xs) else None; t = ys.min() if len(ys) else None
    r = np.nonzero(d[1])[1].max() if d[1].any() else -1
    b = np.nonzero(d[2])[0].max() if d[2].any() else -1
    m = d[3] & (np.arange(w)[None, :] < r)
    if m.any(): b = max(b, np.nonzero(m)[0].max())
    return l, t, r, b

def model(a, f, margin, limits, light, blank):
    h, w = a.shape[:2]
    l, t, r, b = box(a, f)
    L = w if l is None else int(l); T = h if t is None else int(t)
    bw = max(int(r), 0) - L + 1; bh = max(int(b), 0) - T + 1
    crossed = l is not None and (bw <= 0 or bh <= 0)
    if (r < 0 and b < 0 and l is None) or bw <= 0 or bh <= 0: bw = bh = 0
    cw, ch = w - bw, h - bh
    if crossed and os.environ.get("CROSSED_KEEP") == "1": return (0, 0, w, h)
    sh = lambda s, p: math.floor(s * p / 100 + 1e-9)
    if blank is not None and (cw > sh(w, blank) or ch > sh(h, blank)): return "blank"
    if bw == 0 or (cw == 0 and ch == 0): return (0, 0, w, h)
    if limits and not (sh(w, limits[0]) <= cw <= sh(w, limits[1]) and sh(h, limits[0]) <= ch <= sh(h, limits[1])): return (0, 0, w, h)
    if light is not None:
        ls = []
        for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            c = a[y, x] / 255.0; ls.append(50 * (c.max() + c.min()))
        if not sorted(ls)[1] >= light: return (0, 0, w, h)
    if margin is None: m = 0
    elif margin[1]: m = sh(min(w, h), margin[0])
    else: m = margin[0]
    x0, y0 = max(0, L - m), max(0, T - m); x1, y1 = min(w, L + bw + m), min(h, T + bh + m)
    return (x0, y0, x1 - x0, y1 - y0)

imgs = sorted(glob.glob(f"{H}/adv/trim/*.png"))
imgs = [i for i in imgs if not re.search(r"_(1x40|40x1|2x40|40x2)\.png", i)]
def job(s):
    r = random.Random(s); f = r.choice(imgs); a = np.array(Image.open(f).convert("RGB"))
    fz = r.choice([0, 1, 3, 10, 30]); margin = r.choice([None, None, (0, False), (1, False), (5, False), (100, False), (1, True), (10, True), (50, True)])
    limits = r.choice([None, None, (0, 100), (1, 20), (5, 50), (10, 40), (50, 100), (0, 0)]); light = r.choice([None, None, 0, 50, 80, 100]); blank = r.choice([None, None, 0, 1, 50, 99, 100])
    o = ["--lossless", "--trim", "--fuzz", f"{fz}%"]
    if margin: o += ["--trim-margin", f"{margin[0]}{'%' if margin[1] else ''}"]
    if limits: o += ["--trim-limits", f"{limits[0]}%,{limits[1]}%"]
    if light is not None: o += ["--trim-lightness", str(light)]
    if blank is not None: o += ["--skip-blank", str(blank)]
    want = model(a, fz / 100, margin, limits, light, blank)
    d = tempfile.mkdtemp(dir=TD); out = f"{d}/o.avif"
    try:
        p = subprocess.run([BIN, f, out] + o, capture_output=True, text=True, timeout=60)
        if p.returncode == 3 and "blank" in p.stderr: got = "blank"
        elif p.returncode: return (s, "rc", p.returncode, p.stderr[-120:], f, o)
        else:
            c = avt.check(out, d); dec = np.rint(avt.to_rgb(c["si"], c["planes"]) * 255).astype(np.uint8); got = dec.shape[1], dec.shape[0]
            if want != "blank" and got == want[2:]:
                x, y, w, h = want
                if not np.array_equal(dec, a[y:y + h, x:x + w]): return (s, "pixels", want, got, f, o)
                return None
        if got == want: return None
        return (s, "verdict", want, got, f, o)
    finally:
        shutil.rmtree(d, ignore_errors=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
bad = collections.Counter(); ex = []
with cf.ThreadPoolExecutor(3) as tp:
    for r in tp.map(job, range(N)):
        if r: bad[r[1]] += 1; ex.append(r)
print(f"{N - sum(bad.values())}/{N} decisions as the model; {dict(bad)}")
for e in ex[:30]: print(" ", e[1], "want", e[2], "got", e[3], os.path.basename(e[4]), " ".join(e[5][2:]))
# ---- auto-level windows ----
T = TD
a = np.full((80, 120, 3), 128, np.uint8); a[0, 0] = 0; a[79, 119] = 255; a[20:60, 30:90] = np.tile(np.linspace(100, 160, 60).astype(np.uint8)[None, :, None], (40, 1, 3))
wpng(f"{T}/w.png", a)
o = f"{T}/o.avif"
subprocess.run([BIN, f"{T}/w.png", o, "--lossless", "--crop", "60x40+30+20", "--auto-level"], check=True, capture_output=True)
c = avt.check(o); y = c["planes"][0]
print("auto-level inside --crop: output range", y.min(), y.max(), "(want 0..255)")
b = np.full((80, 120, 3), 255, np.uint8); b[20:60, 30:90] = a[20:60, 30:90]
wpng(f"{T}/w2.png", b)
subprocess.run([BIN, f"{T}/w2.png", o, "--lossless", "--trim", "--auto-level"], check=True, capture_output=True)
c = avt.check(o); y = c["planes"][0]
print("auto-level after --trim: size", y.shape, "range", y.min(), y.max(), "(want 40x60, 0..255)")
# ---- alpha stays ----
for f in ("agrad_120x80", "hidden_120x80", "astep_120x80"):
    im = np.array(Image.open(f"{H}/adv/alpha/{f}.png"))
    for op in (["--negate"], ["--level", "10%,90%", "--gamma", "2"], ["--auto-level"], ["--grayscale", "Average"]):
        subprocess.run([BIN, f"{H}/adv/alpha/{f}.png", o, "--lossless"] + op, check=True, capture_output=True)
        c = avt.check(o)
        print(f, op, "alpha unchanged:", np.array_equal(c["aplane"], im[..., 3]))
shutil.rmtree(TD, ignore_errors=True)
