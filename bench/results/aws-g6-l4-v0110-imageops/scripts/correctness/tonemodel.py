#!/usr/bin/env python3
"""the fix head's k_tone at 4:2:0, in numpy (float32), against its own --dump-source output: tests that reading a copy of the chroma gives exactly the
documented result, with no dependence on block order.  Also runs each case 5 times: the planes must be the same every time."""
import glob, os, subprocess, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.expanduser("~/t"))
from trimref_lib import chroma_up
H = os.path.expanduser("~"); B = os.environ.get("BIN", f"{H}/fix/build/av1gpu"); T = "/dev/shm/tm"; os.makedirs(T, exist_ok=True)
f32 = np.float32
def dump(f, o):
    subprocess.run([B, f, f"{T}/o.avif", "--dump-source", f"{T}/d.yuv", "--preset", "turbo3"] + o, check=True, capture_output=True)
    return open(f"{T}/d.yuv", "rb").read()
def tone_v(v, s, o, e, neg):
    if s != 1 or o != 0 or e != 1:
        v = np.clip(v * f32(s) + f32(o), f32(0), f32(1))
        if e != 1: v = np.power(v, f32(e))
    return f32(1) - v if neg else v
def rgb(Y, U, V):
    yy = Y / f32(255); cb = (U - f32(128)) / f32(255); cr = (V - f32(128)) / f32(255)
    return yy + f32(1.402) * cr, yy - f32(0.344136286) * cb - f32(0.714136286) * cr, yy + f32(1.772) * cb
UP = [True]
def model(raw, w, h, s, o, e, neg):
    cw, ch = (w + 1) // 2, (h + 1) // 2
    a = np.frombuffer(raw, np.uint8).astype(np.float32)
    Y = a[:w * h].reshape(h, w); U = a[w * h:w * h + cw * ch].reshape(ch, cw); V = a[w * h + cw * ch:w * h + 2 * cw * ch].reshape(ch, cw)
    Uu = chroma_up(U.astype(np.int64), w, h).astype(np.float32); Vu = chroma_up(V.astype(np.int64), w, h).astype(np.float32)
    Un = np.repeat(np.repeat(U, 2, 0), 2, 1)[:h, :w]; Vn = np.repeat(np.repeat(V, 2, 0), 2, 1)[:h, :w]
    if not UP[0]: Uu, Vu = Un, Vn
    r, g, b = (tone_v(c, s, o, e, neg) for c in rgb(Y, Uu, Vu))
    Yo = np.clip(np.floor((f32(0.299) * r + f32(0.587) * g + f32(0.114) * b) * f32(255) + f32(0.5)), 0, 255)
    r, g, b = (tone_v(c, s, o, e, neg) for c in rgb(Y, Un, Vn))
    cbn = (f32(-0.168735892) * r - f32(0.331264108) * g + f32(0.5) * b) * f32(255) + f32(128)
    crn = (f32(0.5) * r - f32(0.418687589) * g - f32(0.081312411) * b) * f32(255) + f32(128)
    def blk(c):  # mean over the 2x2 block; edge blocks of odd sizes average the padding, which replicates the edge
        c = np.pad(c, ((0, h % 2), (0, w % 2)), mode="edge")
        return c.reshape(ch, 2, cw, 2).mean((1, 3))
    Uo = np.clip(np.floor(blk(cbn) + f32(0.5)), 0, 255); Vo = np.clip(np.floor(blk(crn) + f32(0.5)), 0, 255)
    return Yo, Uo, Vo
cases = {"negate": ([" --negate"], (1, 0, 1, True)), "level": (["--level", "10%,90%"], (1 / 0.8, -0.1 / 0.8, 1, False)), "gamma": (["--gamma", "0.7"], (1, 0, 1 / 0.7, False)),
         "level+gamma+negate": (["--level", "5%,95%,1.3", "--negate"], (1 / 0.9, -0.05 / 0.9, 1 / 1.3, True))}
files = (sorted(glob.glob("/dev/shm/cwe/j/*.jpg")) if os.environ.get("CWEDGE") else []) + sorted(glob.glob(f"{H}/images_std/*.jpg"))[:6] + sorted(glob.glob("/dev/shm/san3/jpg/r*.jpg"))[:12] + [f"{H}/tone2/lineart.png", f"{H}/adv/trim/noise_257x65.png", f"{H}/adv/trim/corners_1_65x33.png"] + sorted(glob.glob("/dev/shm/san3/png/r*.png"))[:12]
for f in files:
    from PIL import Image
    w, h = Image.open(f).size
    if os.environ.get("UPRULE") == "jpeg420":
        im = Image.open(f)
        UP[0] = im.format == "JPEG" and len(im.layer) >= 3 and im.layer[0][1] // im.layer[1][1] == 2 and im.layer[0][2] // im.layer[1][2] == 2 and (w + 1) // 2 > 2
    base = dump(f, [])
    for name, (o, (s, oo, e, neg)) in cases.items():
        o = [x.strip() for x in o]
        outs = {hashlib.sha256(dump(f, o)).hexdigest() for _ in range(5)}
        got = dump(f, o)
        Yo, Uo, Vo = model(base, w, h, s, oo, e, neg)
        cw, ch = (w + 1) // 2, (h + 1) // 2
        g = np.frombuffer(got, np.uint8).astype(int)
        dy = np.abs(g[:w * h].reshape(h, w) - Yo); du = np.abs(g[w * h:w * h + cw * ch].reshape(ch, cw) - Uo); dv = np.abs(g[w * h + cw * ch:].reshape(ch, cw) - Vo)
        print(f"{os.path.basename(f)[:26]:26s} up={int(UP[0])} {name:20s} runs alike: {len(outs) == 1}  Y off by >0: {int((dy > 0).sum())} (max {int(dy.max())})  chroma off: {int((du > 0).sum() + (dv > 0).sum())} (max {int(max(du.max(), dv.max()))})", flush=True)
