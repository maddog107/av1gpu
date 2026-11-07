#!/usr/bin/env python3
"""byte identity of main and the PR without the new options: batch and single-file, across modes and sets.  Prints each
comparison; 'DIFF' lines are findings.  Compares output bytes, the file sets, exit status and the skip messages."""
import glob, hashlib, json, os, random, re, shutil, subprocess, sys, tempfile, time
H = os.path.expanduser("~")
MAIN, PR = f"{H}/main/build/av1gpu", os.environ.get("BIN", f"{H}/pr/build/av1gpu")
W = os.environ.get("IDENTDIR", f"{H}/ident"); shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
def lst(name, files):
    p = f"{W}/{name}.lst"
    open(p, "w").write("".join(f + "\n" for f in files)); return p
g = lambda *ps: sorted(f for p in ps for f in glob.glob(f"{H}/{p}"))
rnd = random.Random(65)
sets = {
    "photos": g("images_std/*.jpg"),
    "sizes": g("images/*.jpg"),
    "comics": g("comics/*.jpg"),
    "gate": g("pr/tests/gate_images/*"),
    "advjpg": g("adv/jpeg/*.jpg"),
    "advpng": g("adv/trim/*.png", "adv/alpha/*.png", "adv/deep/*.png", "adv/flat/*_1*.png", "adv/flat/*_8*.png", "adv/flat/*_64*.png"),
    "webp": g("adv/webp/*.webp"),
    "broken": g("adv/broken/*") + g("adv/jpeg/ex6_420_160x96.jpg"),
}
import numpy as np
sys.path.insert(0, f"{H}/t")
from gen_lib import wpng
LIM = f"{H}/adv/limits"; os.makedirs(LIM, exist_ok=True)
for w_, h_ in [(64, 65472), (64, 65473), (64, 65535), (64, 65536), (64, 65537), (65535, 64), (65536, 64), (65537, 64), (70000, 8)]:
    p_ = f"{LIM}/l{w_}x{h_}.png"
    if not os.path.exists(p_):
        a_ = np.full((h_, w_, 3), 255, np.uint8); a_[h_ // 3: 2 * h_ // 3, w_ // 3: 2 * w_ // 3] = (10, 90, 200); wpng(p_, a_)
sets["limits"] = sorted(glob.glob(f"{LIM}/*.png"))
sets["mixed"] = rnd.sample(sets["photos"], 12) + rnd.sample(sets["advjpg"], 20) + rnd.sample(sets["advpng"], 30) + sets["webp"]
rnd.shuffle(sets["mixed"])
print({k: len(v) for k, v in sets.items()})
modes = {
    "default": [], "fast": ["--preset", "fast"], "turbo3": ["--preset", "turbo3"], "base": ["--preset", "base", "--small-preset", "none"],
    "444": ["--yuv", "444"], "10": ["--depth", "10"], "444_10": ["--yuv", "444", "--depth", "10"],
    "q": ["--qindex", "60,180"], "crop": ["--crop", "100x80+7+3"], "maxsize": ["--max-size", "300x300"], "maxh": ["--max-height", "2960", "--preset", "turbo3"],
    "palon": ["--palette", "on"], "paloff": ["--palette", "off"], "tiles": ["--tiles", "0.01,1,1"], "noalpha": ["--no-alpha"], "aq": ["--alpha-quality", "40"],
    "cpuent": ["--cpu-entropy"], "tunenone": ["--tune", "none"], "maxmp": ["--max-mp", "0.05"],
}
llmodes = {"ll": ["--lossless"], "llyc": ["--lossless", "--ycgco-re"], "llcrop": ["--lossless", "--crop", "50x40+3+5"], "llmax": ["--lossless", "--max-size", "40x40"], "llpal": ["--lossless", "--palette", "on"]}
envs = {"": {}, "ingest1": {"GP_INGEST_DECODER": "1"}}
def tree(d):
    r = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f); r[os.path.relpath(p, d)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return r
def msgs(err):
    l = [x for x in err.splitlines() if "] skipped " in x or "rror" in x and " images (" not in x and "stage ms" not in x]
    return sorted(l)
tot = same = 0; t0 = time.time()
def cmp_run(tag, argv, tmo=1800, env=None):
    global tot, same
    res = []
    for b in (MAIN, PR):
        o = f"{W}/o_{'main' if b == MAIN else 'other'}"; shutil.rmtree(o, ignore_errors=True); os.makedirs(o)
        e = dict(os.environ, **(env or {}))
        a = [x.replace("@OUT", o) for x in argv]
        try:
            r = subprocess.run([b] + a, capture_output=True, text=True, timeout=tmo, env=e)
            res.append((r.returncode, tree(o), msgs(r.stderr)))
        except subprocess.TimeoutExpired:
            res.append(("timeout", {}, []))
    (rc0, t_0, m0), (rc1, t_1, m1) = res
    tot += len(t_0); nsame = sum(1 for k in t_0 if t_1.get(k) == t_0[k]); same += nsame
    m1n = [x.replace(W + "/o_other", W + "/o_main") for x in m1]
    ok = rc0 == rc1 and t_0 == t_1 and m1n == m0
    if not ok:
        with open(os.environ.get("IDENTDIFF", f"{W}/diffs.txt"), "a") as df:
            df.write(f"== {tag}: rc {rc0}/{rc1}\n")
            for k in sorted(set(t_0) | set(t_1)):
                if t_0.get(k) != t_1.get(k): df.write(f"   file {k}: main {'missing' if k not in t_0 else t_0[k][:12]} other {'missing' if k not in t_1 else t_1[k][:12]}\n")
            for x in sorted(set(m0) - set(m1n)): df.write(f"   only main: {x}\n")
            for x in sorted(set(m1n) - set(m0)): df.write(f"   only other: {x}\n")
    print(f"{'same' if ok else 'DIFF'} {tag}: rc {rc0}/{rc1}, {len(t_0)}/{len(t_1)} files, {nsame} equal" + ("" if ok else f"  only-main {sorted(set(t_0) - set(t_1))[:5]} only-pr {sorted(set(t_1) - set(t_0))[:5]} differ {[k for k in t_0 if k in t_1 and t_0[k] != t_1[k]][:5]} msgs0 {m0[:3]} msgs1 {m1[:3]}"), flush=True)
for sn, files in sets.items():
    l = lst(sn, files)
    if sn == "limits":
        for mn, opts in {"default": [], "444": ["--yuv", "444"], "10": ["--depth", "10"], "ll": ["--lossless"], "crop": ["--crop", "64x65500+0+0"], "crop472": ["--crop", "64x65472+0+0"],
                         "maxh": ["--max-height", "65500"], "maxs": ["--max-size", "60000x65473"], "maxmp": ["--max-mp", "1"], "cpuent": ["--cpu-entropy"],
                         "crop2": ["--crop", "64x65472+0+1", "--resize", "64x65472!"]}.items():
            cmp_run(f"batch {sn} {mn}", ["--batch", l, "--outdir", "@OUT", "--preset", "turbo3"] + opts)
            for f in files: cmp_run(f"single {os.path.basename(f)} {mn}", [f, "@OUT/o.avif", "--preset", "turbo3"] + opts)
        continue
    for mn, opts in modes.items():
        cmp_run(f"batch {sn} {mn}", ["--batch", l, "--outdir", "@OUT"] + opts)
    for mn, opts in llmodes.items():
        cmp_run(f"batch {sn} {mn}", ["--batch", l, "--outdir", "@OUT"] + opts)
    cmp_run(f"batch {sn} default ingest1", ["--batch", l, "--outdir", "@OUT"], env={"GP_INGEST_DECODER": "1"})
print("batch part done", time.time() - t0, flush=True)
# single-file: a sample of everything, each with a subset of modes
pool = rnd.sample(sets["photos"], 6) + rnd.sample(sets["sizes"], 6) + rnd.sample(sets["comics"], 3) + sets["gate"][:6] + rnd.sample(sets["advjpg"], 25) + rnd.sample(sets["advpng"], 40) + sets["webp"]
for f in pool:
    tagf = os.path.basename(f)
    for mn in ("default", "444_10", "crop", "maxsize", "palon", "fast"):
        cmp_run(f"single {tagf} {mn}", [f, "@OUT/o.avif"] + modes[mn])
    if not f.endswith(".jpg"):
        for mn in ("ll", "llyc"):
            cmp_run(f"single {tagf} {mn}", [f, "@OUT/o.avif"] + llmodes[mn])
print(f"TOTAL {same}/{tot} outputs equal, {time.time() - t0:.0f} s")
