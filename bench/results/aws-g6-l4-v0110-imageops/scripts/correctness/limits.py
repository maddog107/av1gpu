#!/usr/bin/env python3
"""which sizes write a good file, a broken file, or an error: main and PR, single and batch.  Prints a table; every written file goes through dav1d and aomdec."""
import os, re, shutil, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.expanduser("~/t"))
from gen_lib import wpng
import avt
H = os.path.expanduser("~"); BINS = {"main": f"{H}/main/build/av1gpu", "pr": f"{H}/pr/build/av1gpu", "fix": f"{H}/fix/build/av1gpu", "fix3": f"{H}/fix3/build/av1gpu"}
USE = os.environ.get("LIMBINS", "main,pr").split(",")
OTHER = [b for b in USE if b != "main"][0]
T = "/dev/shm/lim"; shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
def png(w, h, gray=False):
    p = f"{T}/{'g' if gray else 'c'}{w}x{h}.png"
    if not os.path.exists(p):
        a = np.full((h, w, 3), 255, np.uint8)
        a[h // 3: max(h // 3 + 1, 2 * h // 3), w // 3: max(w // 3 + 1, 2 * w // 3)] = (10, 90, 200)
        wpng(p, a[..., 1] if gray else a)
    return p
small = f"{T}/small.png"; wpng(small, np.full((32, 32, 3), 200, np.uint8))
def verdict(path):
    if not os.path.exists(path): return "no file"
    td = tempfile.mkdtemp(dir=T)
    try:
        c = avt.check(path, td)
        return f"GOOD {c['si'][1]}x{c['si'][2]} {c['si'][0]} {c['si'][3]}b"
    except avt.Bad as e:
        return "BROKEN: " + str(e).split("\n")[0][:60]
    finally:
        shutil.rmtree(td, ignore_errors=True)
def tail(e):
    l = [x for x in e.splitlines() if x.strip() and "stage ms" not in x and not x.startswith("[av1gpu] alpha") and " images (" not in x and "trim:" not in x]
    return (l[-1][:105] if l else "")
def single(b, inp, opts):
    o = f"{T}/o.avif"
    if os.path.exists(o): os.remove(o)
    try:
        p = subprocess.run([b, inp, o] + opts, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", ""
    return p.returncode, verdict(o), tail(p.stderr)
def batch(b, inp, opts, lineopt=""):
    l = f"{T}/l"; open(l, "w").write(f"{small}\n{inp}{lineopt}\n{T}/small2.png\n")
    shutil.copy(small, f"{T}/small2.png")
    od = f"{T}/out"; shutil.rmtree(od, ignore_errors=True)
    try:
        p = subprocess.run([b, "--batch", l, "--outdir", od] + opts, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", 0, ""
    n = os.path.basename(inp) + ".avif"
    others = sum(os.path.exists(f"{od}/{x}.avif") for x in ("small.png", "small2.png"))
    return p.returncode, verdict(f"{od}/{n}"), others, tail(p.stderr)
print("| bin | input | options | single rc | single file | batch rc | batch file | others kept | message |"); print("|---|---|---|---|---|---|---|---|---|")
def row(bn, inp, opts, lineopt=""):
    b = BINS[bn]
    r, v, m = single(b, inp, opts + (["--resize", lineopt.split("=", 1)[1]] if lineopt else []))
    r2, v2, ok, m2 = batch(b, inp, opts, ("\tresize=" + lineopt.split("=", 1)[1]) if lineopt else "")
    print(f"| {bn} | {os.path.basename(inp)} | {' '.join(opts + ([lineopt] if lineopt else []))} | {r} | {v} | {r2} | {v2} | {ok}/2 | {(m or m2)} |", flush=True)
# ---- the lead's list, no options ----
sizes = [(64, 65535), (64, 65536), (64, 65537), (65536, 64), (65537, 64)]
for w, h in sizes:
    for bn in USE: row(bn, png(w, h), [])
# ---- the same and the edges in each mode ----
for w, h in [(64, 65472), (64, 65473), (64, 65535), (64, 65536), (64, 65537), (65535, 64), (65536, 64), (65537, 64), (16, 65536), (1, 65536), (65536, 1)]:
    for opts in (["--yuv", "444"], ["--depth", "10"], ["--lossless"]):
        for bn in USE: row(bn, png(w, h), opts)
    for bn in USE: row(bn, png(w, h, gray=True), [])
# ---- --resize, PR only ----
src = f"{H}/adv/trim/bar_mid_200x120.png"
for g in ("1x65535!", "1x65536!", "1x65472!", "1x65473!", "64x65472!", "64x65473!", "64x65535!", "64x65536!", "64x65537!", "65535x1!", "65536x1!", "65537x1!", "65536x64!", "65537x64!", "70000x64!", "262144x1!", "262145x1!", "300000x1!"):
    row(OTHER, src, [], "resize=" + g)
for g in ("1x65472!", "1x65473!", "64x65472!", "64x65473!", "65536x64!"):
    for opts in (["--yuv", "444"], ["--depth", "10"], ["--lossless"]):
        row(OTHER, src, opts, "resize=" + g)
# ---- the transform path without new options, on both: a crop or a shrink ending in a tall frame ----
for opts in (["--crop", "64x65472+0+0"], ["--crop", "64x65473+0+0"], ["--crop", "64x65500+0+0"], ["--max-height", "65472"], ["--max-height", "65473"], ["--max-height", "65500"]):
    for bn in USE: row(bn, png(64, 65535), opts)
    for bn in USE: row(bn, png(64, 65537), opts)
for opts in (["--cpu-entropy"],):
    for w, h in [(64, 65536), (65536, 64)]:
        for bn in USE: row(bn, png(w, h), opts)
for opts in (["--crop", "64x65000+0+0", "--resize", "64x65535!"], ["--crop", "64x65472+0+1", "--resize", "64x65472!"]):
    for bn in USE: row(bn, png(64, 65535), opts)
print("LIMITS_DONE")
