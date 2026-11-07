#!/usr/bin/env python3
"""batch behaviour under pressure.  Each input i carries its own colour (block), own size, and JPEGs their own ICC profile and EXIF orientation; after a run
every output is decoded and matched to its input."""
import glob, hashlib, io, json, os, re, shutil, struct, subprocess, sys, tempfile, threading, time, random, collections
import concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
from gen_lib import wpng, chunk
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAIN = f"{H}/main/build/av1gpu"
W = f"{H}/bp"; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
fails = []
def F(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ((": " + str(detail)) if detail else ""), flush=True)
    if not ok: fails.append(name)

def exif_bytes(o):
    t = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x112, 3, 1, o) + struct.pack("<I", 0)
    return b"Exif\x00\x00" + t

def fake_icc(i):
    """a syntactically plausible ICC profile, different for each image"""
    body = bytearray(600)
    body[0:4] = struct.pack(">I", 600); body[12:16] = b"mntr"; body[16:20] = b"RGB "; body[20:24] = b"XYZ "; body[36:40] = b"acsp"
    body[128:132] = struct.pack(">I", 0); body[132:] = hashlib.sha256(str(i).encode()).digest() * 13 + b"\0" * (600 - 132 - 416)
    return bytes(body[:600])

colr = lambda i: tuple(int(v) for v in (30 + (i * 37) % 180, 30 + (i * 73) % 180, 30 + (i * 151) % 180))

def mk(i, kind, d):
    """image i: white margins, a block of colour i.  Returns dict(path, w, h, box, rgb, orient, icc)"""
    w, h = 64 + (i * 13) % 200, 48 + (i * 29) % 160
    a = np.full((h, w, 3), 255, np.uint8)
    mx, my = 5 + i % 17, 3 + i % 13
    bw, bh = w - 2 * mx - i % 7, h - 2 * my - i % 5
    a[my:my + bh, mx:mx + bw] = colr(i)
    m = dict(w=w, h=h, box=(mx, my, bw, bh), rgb=colr(i), orient=1, icc=None, kind=kind)
    if kind == "jpeg":
        o = 1 + i % 8; icc = fake_icc(i)
        p = f"{d}/{i:05d}.jpg"; Image.fromarray(a).save(p, quality=98, subsampling=0, exif=exif_bytes(o), icc_profile=icc)
        m.update(orient=o, icc=icc)
    elif kind == "jpeg420":
        o = 1 + i % 8
        p = f"{d}/{i:05d}.jpg"; Image.fromarray(a).save(p, quality=98, subsampling=2, exif=exif_bytes(o), icc_profile=fake_icc(i))
        m.update(orient=o, icc=fake_icc(i))
    elif kind == "png": p = f"{d}/{i:05d}.png"; wpng(p, a)
    elif kind == "rgba":
        al = np.zeros((h, w), np.uint8); al[my:my + bh, mx:mx + bw] = 255
        p = f"{d}/{i:05d}.png"; wpng(p, np.dstack([a, al]))
    elif kind == "webp": p = f"{d}/{i:05d}.webp"; Image.fromarray(a).save(p, lossless=True)
    elif kind == "gray":
        g = a.mean(2).astype(np.uint8); p = f"{d}/{i:05d}.jpg"; Image.fromarray(g).save(p, quality=98); m["rgb"] = None
    elif kind == "bjpeg":  # blank
        p = f"{d}/{i:05d}.jpg"; Image.fromarray(np.full((h, w, 3), 255, np.uint8)).save(p, quality=95); m["blank"] = True
    elif kind == "bpng":
        p = f"{d}/{i:05d}.png"; wpng(p, np.full((h, w, 3), 250, np.uint8)); m["blank"] = True
    elif kind == "bwebp":
        p = f"{d}/{i:05d}.webp"; Image.fromarray(np.full((h, w, 3), 255, np.uint8)).save(p, lossless=True); m["blank"] = True
    elif kind == "brgba":
        p = f"{d}/{i:05d}.png"; wpng(p, np.zeros((h, w, 4), np.uint8)); m["blank"] = True
    m["path"] = p
    return m

def verify(m, out, opts):
    """decoded output against its own input: size, the block's colour, ICC, orientation"""
    c = avt.check(out)
    fmt, w, h, depth = c["si"][:4]
    errs = []
    if not m.get("blank"):
        mx, my, bw, bh = m["box"]
        if "--trim" in opts and (w, h) != (bw, bh) and abs(w - bw) > 2: errs.append(f"size {w}x{h} want {bw}x{bh}")
        if m["rgb"] is not None and "--grayscale" not in opts:
            rgb = avt.to_rgb(c["si"], c["planes"]); hh, ww = rgb.shape[:2]
            cen = rgb[hh // 2 - 1:hh // 2 + 1, ww // 2 - 1:ww // 2 + 1].mean((0, 1)) * 255
            if np.abs(cen - np.array(m["rgb"])).max() > 8: errs.append(f"colour {cen.round()} want {m['rgb']}")
    if m["icc"] is not None and c["icc"] != m["icc"]: errs.append("ICC of another image" if c["icc"] else "ICC lost")
    if m["icc"] is None and c["icc"]: errs.append("unexpected ICC")
    want = {1: (None, None), 2: (None, 1), 3: (2, None), 4: (None, 0), 5: (1, 0), 6: (3, None), 7: (3, 0), 8: (1, None)}[m["orient"]]
    if (c["irot"], c["imir"]) != want: errs.append(f"orientation {(c['irot'], c['imir'])} want {want} (EXIF {m['orient']})")
    return errs

def batch(name, metas, opts=(), to=600, env=None, extra=None, bin_=None):
    lst = f"{W}/{name}.lst"
    ls = [m["path"] for m in metas] + (extra or [])
    open(lst, "w").write("".join(x + "\n" for x in ls))
    od = f"{W}/{name}.out"; shutil.rmtree(od, ignore_errors=True)
    t = time.time()
    try:
        p = subprocess.run([bin_ or BIN, "--batch", lst, "--outdir", od, "--stats"] + list(opts), capture_output=True, text=True, timeout=to, env=dict(os.environ, **(env or {})))
    except subprocess.TimeoutExpired:
        F(f"{name}: no hang", False, f"timeout {to} s"); return None
    return p, od, time.time() - t

def check_all(name, metas, opts, p, od, blanks_ok=True):
    ok = True; bad = []
    have = set(os.listdir(od)) if os.path.isdir(od) else set()
    want_written = [m for m in metas if not (m.get("blank") and "--skip-blank" in opts)]
    for m in metas:
        n = os.path.basename(m["path"]) + ".avif"
        if m.get("blank") and "--skip-blank" in opts:
            if n in have: bad.append((n, "blank written"))
            if f"skipped {m['path']}: blank" not in p.stderr: bad.append((n, "no blank message"))
        elif n not in have: bad.append((n, "missing"))
        else:
            try:
                for e in verify(m, f"{od}/{n}", opts): bad.append((n, e))
            except avt.Bad as e: bad.append((n, str(e)))
    F(f"{name}: rc 0, {len(have)} files, all match their inputs", p.returncode == 0 and not bad, f"rc {p.returncode} {bad[:4]} {len(bad)} {p.stderr[-200:] if p.returncode else ''}")

# ---- A: a blank image at every position ----
d = f"{W}/a"; os.makedirs(d)
base = [mk(i, k, d) for i, k in enumerate(["jpeg", "png", "jpeg420", "rgba", "webp", "jpeg", "gray", "png"])]
for kind in ("bjpeg", "bpng", "bwebp", "brgba"):
    for pos in range(len(base) + 1):
        dd = f"{W}/a_{kind}_{pos}"; os.makedirs(dd)
        b = mk(1000 + pos, kind, dd)
        l = base[:pos] + [b] + base[pos:]
        r = batch(f"a_{kind}_{pos}", l, ["--trim", "--fuzz", "5%", "--skip-blank", "99", "--small-preset", "none", "--preset", "turbo3"])
        if r: check_all(f"A {kind} at {pos}", l, ["--trim", "--fuzz", "5%", "--skip-blank", "99"], *r)

# ---- B: whole lists of blanks ----
for kind in ("bjpeg", "bpng", "bwebp", "brgba"):
    for n in (1, 2, 3, 7, 48, 200, 1000):
        dd = f"{W}/b_{kind}_{n}"; os.makedirs(dd)
        l = [mk(i, kind, dd) for i in range(n)]
        for env in ({}, {"GP_ENCODE_IMGS": "1"}):
            r = batch(f"b_{kind}_{n}", l, ["--trim", "--skip-blank", "99"], env=env)
            if r:
                p, od, dt = r
                nf = len(os.listdir(od)) if os.path.isdir(od) else 0
                m = re.search(r"(\d+) skipped as blank", p.stderr)
                F(f"B {kind} x{n} {env}: rc 0, no files, count", p.returncode == 0 and nf == 0 and m and int(m[1]) == n, f"rc {p.returncode} files {nf} blank {m and m[1]} {dt:.1f}s {p.stderr[-200:] if p.returncode else ''}")
# mixed blanks of all kinds
dd = f"{W}/bmix"; os.makedirs(dd)
l = [mk(i, ["bjpeg", "bpng", "bwebp", "brgba"][i % 4], dd) for i in range(400)]
r = batch("bmix", l, ["--trim", "--skip-blank", "99"])
if r:
    p, od, dt = r; F(f"B mixed blanks x400", p.returncode == 0 and not (os.path.isdir(od) and os.listdir(od)), f"rc {p.returncode} {dt:.1f}s")

# ---- E: mixed formats, trim + everything ----
d = f"{W}/e"; os.makedirs(d)
rnd = random.Random(5)
kinds = ["jpeg", "jpeg420", "png", "rgba", "webp", "gray"]
l = [mk(i, rnd.choice(kinds + ["bjpeg", "bpng"]), d) for i in range(300)]
for tag, opts in [("420", ["--trim", "--fuzz", "5%"]), ("444", ["--trim", "--fuzz", "5%", "--yuv", "444"]), ("10", ["--trim", "--fuzz", "5%", "--depth", "10"]), ("444_10", ["--trim", "--fuzz", "5%", "--yuv", "444", "--depth", "10"]),
                  ("skip", ["--trim", "--fuzz", "5%", "--skip-blank", "99"]), ("skip10", ["--trim", "--fuzz", "5%", "--skip-blank", "99", "--yuv", "444", "--depth", "10"]),
                  ("tone", ["--trim", "--fuzz", "5%", "--skip-blank", "99", "--level", "0%,100%,1.01"]), ("gray", ["--trim", "--fuzz", "5%", "--skip-blank", "99", "--grayscale", "Rec709Luma"]),
                  ("margin", ["--trim", "--fuzz", "5%", "--trim-margin", "0"])]:
    r = batch(f"e_{tag}", l, opts)
    if r: check_all(f"E mixed {tag}", l, opts, *r)
    for en, env in [("1img", {"GP_ENCODE_IMGS": "1"}), ("smallq", {"GP_FALLBACK_QUEUE_MP": "1", "GP_ENCODE_MP": "1"}), ("cpuj", {"GP_INGEST_DECODER": "1"})]:
        if tag in ("skip", "skip10"):
            r = batch(f"e_{tag}_{en}", l, opts, env=env)
            if r: check_all(f"E mixed {tag} {en}", l, opts, *r)

# ---- F: broken files in lists ----
brk = sorted(glob.glob(f"{H}/adv/broken/*"))
good = [m for m in l if m["kind"] in ("jpeg", "png", "rgba")][:40]
r = batch("f_broken", good, ["--trim", "--skip-blank", "99", "--auto-level"], extra=brk)
if r:
    p, od, dt = r
    s = [b for b in brk if f"skipped {b}" not in p.stderr]
    F("F broken files: skipped with a message", p.returncode == 0 and not s, f"rc {p.returncode} not reported {s} {p.stderr[-300:] if p.returncode else ''}")
    # a list line with a bad geometry
ls = [m["path"] for m in good[:5]] + [good[5]["path"] + "\tresize=abc", good[6]["path"] + "\tresize=0.5", good[7]["path"] + "\tresize=1x100000!", good[8]["path"] + "\tresize=100000x1!"] + [m["path"] for m in good[9:15]]
open(f"{W}/g.lst", "w").write("\n".join(ls) + "\n")
od = f"{W}/g.out"
p = subprocess.run([BIN, "--batch", f"{W}/g.lst", "--outdir", od], capture_output=True, text=True, timeout=300)
nf = len(os.listdir(od)) if os.path.isdir(od) else 0
F("G one line with an unusable geometry does not cost the rest of the list", p.returncode == 0 and nf >= 11, f"rc {p.returncode} files {nf}/15: {[x[:150] for x in p.stderr.splitlines() if 'stage ms' not in x][-4:]}")
print("FAILS", len(fails), fails[:40])
