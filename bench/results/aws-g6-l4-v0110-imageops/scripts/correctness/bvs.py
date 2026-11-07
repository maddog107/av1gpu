#!/usr/bin/env python3
"""the same image through --batch and single-file mode: same bytes?  (batches group frames of different sizes, so a per-frame mix-up in a launch shows up here)"""
import glob, hashlib, os, random, shutil, subprocess, sys, tempfile, collections
import concurrent.futures as cf
H = os.path.expanduser("~"); PR, MAIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"), f"{H}/main/build/av1gpu"
T = tempfile.mkdtemp(prefix="bvs_", dir="/dev/shm")
r = random.Random(9)
pool = r.sample(sorted(glob.glob(f"{H}/adv/rand/*")), 70) + r.sample(sorted(glob.glob(f"{H}/adv/jpeg/ex*.jpg")), 20) + r.sample(sorted(glob.glob(f"{H}/adv/trim/*.png")), 20) + r.sample(sorted(glob.glob(f"{H}/adv/alpha/*.png")), 10) + sorted(glob.glob(f"{H}/adv/webp/*.webp"))[:6]
opts = {"none": [], "trim": ["--trim", "--fuzz", "5%"], "trim444": ["--trim", "--fuzz", "3%", "--yuv", "444"], "trim10": ["--trim", "--fuzz", "3%", "--depth", "10", "--trim-margin", "2"],
        "tone": ["--auto-level", "--level", "5%,95%,1.1", "--negate"], "gray": ["--grayscale", "Lightness"], "rs": ["--resize", "77x77^"], "all": ["--trim", "--fuzz", "5%", "--auto-level", "--gamma", "1.2", "--resize", "150%", "--grayscale", "RMS", "--skip-blank", "99"],
        "ll": ["--lossless", "--trim", "--negate"], "llyc": ["--lossless", "--ycgco-re", "--trim", "--fuzz", "3%", "--gamma", "0.7"]}
open(f"{T}/l", "w").write("\n".join(pool) + "\n")
hs = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def single(a):
    b, f, o, k = a
    out = f"{T}/s_{k}/{os.path.basename(f)}.avif"
    p = subprocess.run([b, f, out] + o, capture_output=True, text=True, timeout=120)
    return f, p.returncode, (hs(out) if os.path.exists(out) else None)
def go(tag, b, o):
    d = f"{T}/b_{tag}"; shutil.rmtree(d, ignore_errors=True); sd = f"{T}/s_{tag}"; shutil.rmtree(sd, ignore_errors=True); os.makedirs(sd)
    p = subprocess.run([b, "--batch", f"{T}/l", "--outdir", d] + o, capture_output=True, text=True, timeout=900)
    with cf.ThreadPoolExecutor(3) as tp:
        rs = list(tp.map(single, [(b, f, o, tag) for f in pool]))
    same = diff = onlyb = onlys = neither = 0; ex = []
    for f, rc, h in rs:
        bo = f"{d}/{os.path.basename(f)}.avif"
        hb = hs(bo) if os.path.exists(bo) else None
        if h is None and hb is None: neither += 1
        elif h is None: onlyb += 1; ex.append(("batch only", os.path.basename(f), rc))
        elif hb is None: onlys += 1; ex.append(("single only", os.path.basename(f)))
        elif h == hb: same += 1
        else: diff += 1; ex.append(("differ", os.path.basename(f)))
    print(f"{tag}: batch rc {p.returncode}: same {same}, differ {diff}, only batch {onlyb}, only single {onlys}, neither {neither}  {ex[:5]}", flush=True)
go("main_none", MAIN, [])
for k, o in opts.items(): go(k, PR, o)
shutil.rmtree(T, ignore_errors=True)
