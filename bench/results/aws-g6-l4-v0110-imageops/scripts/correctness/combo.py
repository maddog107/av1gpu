#!/usr/bin/env python3
"""random combinations of the stages against the equivalent magick command in the PR's fixed order: output size and pixels.  JPEG inputs with EXIF too (the displayed picture)."""
import glob, os, random, re, subprocess, sys, tempfile, shutil, collections
import concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
TD = tempfile.mkdtemp(prefix="cmb_", dir="/dev/shm")
pngs = [f for f in glob.glob(f"{H}/adv/trim/*.png") if re.search(r"(bar_|corners_|hgrad|vgrad|noise_|ring_|pxcorners)", f)] + sorted(glob.glob(f"{H}/pr/tests/gate_images/*.png"))
jpgs = sorted(glob.glob(f"{H}/adv/jpeg/ex*_444_*.jpg")) + sorted(glob.glob(f"{H}/adv/jpeg/ex*_420_16*.jpg")) + sorted(glob.glob(f"{H}/images_std/*.jpg"))[:15]
def geo(r):
    W, Hh = r.choice([17, 40, 64, 100, 150, 256]), r.choice([17, 40, 64, 100, 150, 256])
    return r.choice([f"{W}x", f"x{Hh}", f"{W}x{Hh}", f"{W}x{Hh}!", f"{W}x{Hh}^", f"{W}x{Hh}>", f"{W}x{Hh}<", f"{r.choice([25, 50, 75, 125, 150, 200])}%", f"{r.choice([50, 150])}%x{r.choice([50, 150])}%", f"@{r.choice([500, 2000, 8000])}"])
def job(s):
    r = random.Random(s)
    kind = r.choice(["ll", "444", "444", "jpg"])
    f = r.choice(jpgs if kind == "jpg" else pngs)
    av = ["--lossless"] if kind == "ll" else ["--yuv", "444", "--qindex", "12", "--small-preset", "none"]
    im = []
    if kind == "jpg": im += ["-auto-orient"]
    W_, H_ = Image.open(f).size
    if r.random() < .3 and kind != "jpg":
        cw, ch, cx, cy = r.randint(8, W_) if W_ >= 8 else W_, r.randint(8, H_) if H_ >= 8 else H_, r.randint(0, max(0, W_ // 3)), r.randint(0, max(0, H_ // 3))
        av += ["--crop", f"{cw}x{ch}+{cx}+{cy}"]; im += ["-crop", f"{cw}x{ch}+{cx}+{cy}", "+repage"]
    trim = r.random() < .5
    if trim:
        fz = r.choice([0, 3, 10, 30]) if kind != "jpg" else r.choice([10, 30])
        av += ["--trim", "--fuzz", f"{fz}%"]; im += ["-fuzz", f"{fz}%", "-trim", "+repage"]
    if r.random() < .3: av += ["--auto-level"]; im += ["-auto-level"]
    if r.random() < .3:
        l = r.choice(["10%,90%", "5%,95%,1.2", "20%,80%,0.8"]); av += ["--level", l]; im += ["-level", l]
    if r.random() < .3:
        g = r.choice(["0.7", "1.4", "2"]); av += ["--gamma", g]; im += ["-gamma", g]
    if r.random() < .3: av += ["--negate"]; im += ["-negate"]
    g2 = None
    if r.random() < .7:
        g2 = geo(r); av += ["--resize", g2]; im += ["-resize", g2]
    gray = None
    if r.random() < .3:
        gray = r.choice(["Rec601Luma", "Rec709Luma", "Average", "Brightness", "Lightness", "RMS"]); av += ["--grayscale", gray]; im += ["-grayscale", gray]
    d = tempfile.mkdtemp(dir=TD); o, m = f"{d}/o.avif", f"{d}/m.png"
    try:
        p = subprocess.run([BIN, f, o, "--max-mp", "50"] + av, capture_output=True, text=True, timeout=120)
        q = subprocess.run([MAG, f] + im + ["-depth", "8", f"png:{m}"], capture_output=True, text=True, env=env)
        if q.returncode or not os.path.exists(m): return ("magickfail", s, f, av, q.stderr[:120])
        a = np.array(Image.open(m).convert("RGB")).astype(float)
        if p.returncode: return ("rc", s, f, av, p.stderr[-120:])
        c = avt.check(o, d)
        rgb = avt.to_rgb(c["si"], c["planes"]) * 255
        dd = avt.apply_transforms(rgb, c["irot"], c["imir"])
        if dd.shape != a.shape: return ("size", s, f, av, (dd.shape[:2], a.shape[:2]))
        e = np.abs(dd - a); me = float(e.mean()); mx = float(e.max())
        return ("pix", s, f, av, (round(me, 2), round(mx, 1)))
    except avt.Bad as ex:
        return ("bad", s, f, av, str(ex)[:120])
    finally:
        shutil.rmtree(d, ignore_errors=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
res = collections.Counter(); bad = []; hist = []
with cf.ThreadPoolExecutor(3) as tp:
    for r in tp.map(job, range(N)):
        res[r[0]] += 1
        if r[0] == "pix":
            hist.append(r[4][0])
            if r[4][0] > 3: bad.append(r)
        else: bad.append(r)
print(dict(res), "mean diff percentiles", np.percentile(hist, [50, 90, 99, 100]).round(2))
def cls(b):
    av = b[3]
    if "--trim" not in av: return "no trim"
    im = []
    if "--crop" in av: im += ["-crop", av[av.index("--crop") + 1], "+repage"]
    z = av[av.index("--fuzz") + 1]
    q = subprocess.run([MAG, b[2], "-auto-orient"] + im + ["-fuzz", z, "-format", "%@", "info:"], capture_output=True, text=True, env=env).stdout
    m = re.match(r"(\d+)x(\d+)", q)
    return "magick's box is empty" if m and (m[1] == "0" or m[2] == "0") else "OTHER"
c2 = collections.Counter()
for b in bad:
    k = cls(b); c2[(b[0], k)] += 1
    if k != "magick's box is empty": print(b[0], k, os.path.basename(b[2]), " ".join(b[3]), b[4])
print(dict(c2))
shutil.rmtree(TD, ignore_errors=True)
