#!/usr/bin/env python3
"""adversarial inputs into ~/adv/<class>/.  Everything is seeded and reproducible."""
import os, struct, zlib, subprocess, sys, io
import numpy as np
from PIL import Image, TiffImagePlugin
H = os.path.expanduser("~"); A = f"{H}/adv"
rng = np.random.default_rng(65)

def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
def wpng(path, a, ct=None, bits=8, extra=b""):
    a = np.asarray(a)
    if a.ndim == 2: a2 = a; c = 1
    else: a2 = a; c = a.shape[2]
    if ct is None: ct = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    dt = ">u2" if bits == 16 else "u1"
    h, w = a2.shape[:2]
    raw = b"".join(b"\x00" + a2[y].astype(dt).tobytes() for y in range(h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ct, 0, 0, 0)) + extra +
                           chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))

def rgb(w, h, c): return np.full((h, w, 3), c, np.uint8)

# ---- flat images: every size pair of a grid ----
sizes = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
for w in sizes:
    for h in sizes:
        if (w, h).__hash__() % 3 == 0 or w == h or w in (1, 8, 64) or h in (1, 8, 64):
            wpng(f"{A}/flat/white_{w}x{h}.png", rgb(w, h, 255))
for name, c in [("black", (0, 0, 0)), ("red", (255, 0, 0)), ("gray", (128, 128, 128)), ("rand", (37, 201, 90))]:
    for w, h in [(1, 1), (1, 9), (9, 1), (8, 8), (16, 16), (17, 33), (64, 64), (65, 63), (256, 256), (257, 129)]:
        wpng(f"{A}/flat/{name}_{w}x{h}.png", rgb(w, h, c))

# ---- trim content ----
def page(w, h, bg=255): return rgb(w, h, bg).copy()
T = f"{A}/trim"
# a single dark pixel at every corner / edge midpoint / centre
for w, h in [(64, 64), (65, 63), (17, 33), (8, 8), (9, 7), (128, 3), (3, 128), (2, 2), (3, 3), (1, 40), (40, 1), (2, 40), (40, 2)]:
    for tag, (x, y) in dict(tl=(0, 0), tr=(w - 1, 0), bl=(0, h - 1), br=(w - 1, h - 1), t=(w // 2, 0), b=(w // 2, h - 1), l=(0, h // 2),
                             r=(w - 1, h // 2), c=(w // 2, h // 2)).items():
        a = page(w, h); a[y, x] = 0
        wpng(f"{T}/pix_{tag}_{w}x{h}.png", a)
# bars touching exactly one edge
for w, h in [(200, 120), (201, 121), (64, 64)]:
    for tag, (x0, x1, y0, y1) in dict(top=(40, 90, 0, 30), bottom=(40, 90, h - 30, h), left=(0, 30, 30, 70), right=(w - 30, w, 30, 70),
                                       tl=(0, 20, 0, 20), tr=(w - 20, w, 0, 20), bl=(0, 20, h - 20, h), br=(w - 20, w, h - 20, h),
                                       mid=(50, 53, 40, 43), rowonly=(10, w - 10, 30, 31), colonly=(30, 31, 10, h - 10),
                                       col0=(0, 1, 10, h - 10), row0=(10, w - 10, 0, 1), colL=(w - 1, w, 10, h - 10), rowL=(10, w - 10, h - 1, h)).items():
        a = page(w, h); a[y0:y1, x0:x1] = (30, 60, 200)
        wpng(f"{T}/bar_{tag}_{w}x{h}.png", a)
# corners that differ
for w, h in [(200, 120), (65, 33)]:
    for k in range(8):
        cs = rng.integers(0, 256, (4, 3))
        if k < 4:  cs[(k + 1) % 4] = cs[k]           # two corners alike
        a = page(w, h); a[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 90
        a[:h // 8, :w // 8] = cs[0]; a[:h // 8, -w // 8:] = cs[1]; a[-h // 8:, :w // 8] = cs[2]; a[-h // 8:, -w // 8:] = cs[3]
        wpng(f"{T}/corners_{k}_{w}x{h}.png", a)
# one-pixel corners
for k in range(6):
    a = page(97, 61); a[20:40, 30:70] = (10, 120, 30)
    cs = rng.integers(0, 256, (4, 3)); a[0, 0] = cs[0]; a[0, -1] = cs[1]; a[-1, 0] = cs[2]; a[-1, -1] = cs[3]
    wpng(f"{T}/pxcorners_{k}.png", a)
# gradients and splits
for w, h in [(256, 64), (64, 256), (257, 65), (1024, 16)]:
    x = np.tile(np.linspace(0, 255, w).astype(np.uint8)[None, :, None], (h, 1, 3)); wpng(f"{T}/hgrad_{w}x{h}.png", x)
    y = np.tile(np.linspace(0, 255, h).astype(np.uint8)[:, None, None], (1, w, 3)); wpng(f"{T}/vgrad_{w}x{h}.png", y)
    a = page(w, h); a[:, : w // 2] = 0; wpng(f"{T}/vsplit_{w}x{h}.png", a)
    a = page(w, h); a[: h // 2] = 0; wpng(f"{T}/hsplit_{w}x{h}.png", a)
    a = page(w, h, 0); a[1::2, ::2] = 255; a[::2, 1::2] = 255; wpng(f"{T}/checker_{w}x{h}.png", a)
    wpng(f"{T}/noise_{w}x{h}.png", rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
# fuzz boundaries: border colour with near colours at given distance (RGB steps) in a margin ring
for d in [1, 2, 3, 12, 13, 36, 37, 38, 63, 64, 128]:
    a = page(120, 80); a[20:60, 30:90] = 255 - d; a[35:45, 55:65] = 0
    wpng(f"{T}/ring_gray{d}.png", a)
    a = page(120, 80); a[20:60, 30:90] = (255 - d, 255, 255); a[35:45, 55:65] = 0
    wpng(f"{T}/ring_red{d}.png", a)
# a light grey bottom-left corner (the PR's rule 4) with bands
for k, (bl, br, band, tr) in enumerate([((230, 230, 230), (255, 255, 255), (205, 205, 205), (255, 255, 255)),
                                        ((200, 200, 200), (224, 224, 224), (172, 172, 172), (255, 255, 255)),
                                        ((200, 200, 200), (216, 216, 216), (176, 176, 176), (255, 255, 255))]):
    a = page(160, 240); a[20:120, 30:130] = 30
    a[180:181, 10:150] = band; a[200:, :] = bl; a[-1, -1] = br; a[-1, -3:] = br; a[0, -1] = tr
    wpng(f"{T}/rule4_{k}.png", a)
    a = page(160, 240); a[20:120, 30:60] = 30
    a[150:152, 20:50] = band; a[200:, :] = bl; a[-1, -3:] = br
    wpng(f"{T}/rule4b_{k}.png", a)

# ---- transparency ----
P = f"{A}/alpha"
for w, h in [(120, 80), (121, 81), (64, 64), (33, 17)]:
    c = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)     # random colour under alpha 0
    al = np.zeros((h, w), np.uint8); al[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    wpng(f"{P}/hidden_{w}x{h}.png", np.dstack([c, al]))
    b = np.zeros((h, w, 3), np.uint8); b[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = (250, 40, 40)
    wpng(f"{P}/onblack_{w}x{h}.png", np.dstack([b, al]))
    al2 = (np.tile(np.linspace(0, 255, w)[None, :], (h, 1))).astype(np.uint8)
    wpng(f"{P}/agrad_{w}x{h}.png", np.dstack([c, al2]))
    wpng(f"{P}/opaque_{w}x{h}.png", np.dstack([c, np.full((h, w), 255, np.uint8)]))
    wpng(f"{P}/allclear_{w}x{h}.png", np.dstack([c, np.zeros((h, w), np.uint8)]))
    wpng(f"{P}/la_{w}x{h}.png", np.dstack([c[..., 0], al]))
    wpng(f"{P}/rgba16_{w}x{h}.png", np.dstack([c.astype(np.uint16) * 257, al.astype(np.uint16) * 257]), bits=16)
    wpng(f"{P}/la16_{w}x{h}.png", np.dstack([c[..., 0].astype(np.uint16) * 257, al.astype(np.uint16) * 257]), bits=16)
    # alpha steps: small differences of alpha around an opaque middle
    al3 = np.full((h, w), 250, np.uint8); al3[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 255
    wpng(f"{P}/astep_{w}x{h}.png", np.dstack([np.full((h, w, 3), 200, np.uint8), al3]))
# tRNS palette
pal = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
idx = np.zeros((40, 50), np.uint8); idx[10:30, 10:40] = 1; idx[15:25, 20:30] = 2
wpng(f"{P}/trns.png", idx, ct=3, extra=chunk(b"PLTE", pal.tobytes()) + chunk(b"tRNS", bytes([0, 255, 128, 255])))
wpng(f"{P}/pal.png", idx, ct=3, extra=chunk(b"PLTE", pal.tobytes()))

# ---- 16-bit, gray ----
D = f"{A}/deep"
for w, h in [(120, 80), (65, 33)]:
    g = np.tile(np.linspace(0, 65535, w)[None, :], (h, 1)).astype(np.uint16)
    wpng(f"{D}/g16_{w}x{h}.png", g, bits=16)
    wpng(f"{D}/rgb16_{w}x{h}.png", np.dstack([g, g[::-1], np.tile(np.linspace(65535, 0, h)[:, None], (1, w)).astype(np.uint16)]), bits=16)
    pg = np.full((h, w), 65535, np.uint16); pg[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 21845
    wpng(f"{D}/page16_{w}x{h}.png", np.dstack([pg, pg, pg]), bits=16)
    v10 = (rng.integers(0, 1024, (h, w, 3)) / 1023.0 * 65535.0 + 0.5).astype(np.uint16)
    wpng(f"{D}/ten16_{w}x{h}.png", v10, bits=16)
    m = np.full((h, w), 255, np.uint8); m[10:30, 20:40] = 77
    wpng(f"{D}/g8_{w}x{h}.png", m)
    wpng(f"{D}/page8x257_{w}x{h}.png", np.dstack([m, m, m]).astype(np.uint16) * 257, bits=16)
    # 1-bit gray
    bit = np.zeros((h, (w + 7) // 8), np.uint8); bit[:, :] = 0xF0
    wpng(f"{D}/b1_{w}x{h}.png", bit, ct=0, bits=8) if False else None

# ---- JPEG family ----
J = f"{A}/jpeg"; os.makedirs(J, exist_ok=True)
def asym(w, h):
    """an asymmetric picture on white: an L of colours, so every orientation looks different, with a white margin"""
    a = page(w, h); m = max(2, min(w, h) // 8)
    a[m:h - m, m:m + (w - 2 * m) // 3] = (200, 30, 30)                 # left bar red
    a[h - m - (h - 2 * m) // 4:h - m, m:w - m] = (30, 30, 200)         # bottom bar blue
    a[m:m + (h - 2 * m) // 4, w - m - (w - 2 * m) // 4: w - m] = (30, 180, 30)   # top-right block green
    return a
def exif_bytes(o):
    # little-endian TIFF with one IFD entry (Orientation)
    t = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x112, 3, 1, o) + struct.pack("<I", 0)
    return b"Exif\x00\x00" + t
for w, h in [(160, 96), (161, 97), (64, 48), (48, 64)]:
    im = Image.fromarray(asym(w, h))
    for ss, sn in [(0, "444"), (2, "420"), (1, "422")]:
        for o in range(0, 9):
            kw = dict(quality=97, subsampling=ss)
            if o: kw["exif"] = exif_bytes(o)
            im.save(f"{J}/ex{o}_{sn}_{w}x{h}.jpg", **kw)
    im.save(f"{A}/jpeg/prog_{w}x{h}.jpg", quality=92, progressive=True)
    im.convert("L").save(f"{J}/gray_{w}x{h}.jpg", quality=95)
    im.convert("CMYK").save(f"{J}/cmyk_{w}x{h}.jpg", quality=95)
    ppm = f"{J}/_t.ppm"; im.save(ppm)
    for tag, s in [("440", "1x2,1x1,1x1"), ("411", "4x1,1x1,1x1"), ("410", "4x2,1x1,1x1"), ("441", "1x4,1x1,1x1"), ("422r", "2x1,1x1,1x1"),
                   ("420w", "2x2,1x2,1x2")]:
        r = subprocess.run(["cjpeg", "-quality", "95", "-sample", s, "-outfile", f"{J}/s{tag}_{w}x{h}.jpg", ppm], capture_output=True)
        if r.returncode: print("cjpeg", tag, r.stderr[:80])
    gpgm = f"{J}/_t.pgm"; im.convert("L").save(gpgm)
    for tag, s in [("g2x2", "2x2"), ("g2x1", "2x1"), ("g1x2", "1x2"), ("g4x1", "4x1")]:
        r = subprocess.run(["cjpeg", "-quality", "95", "-sample", s, "-outfile", f"{J}/{tag}_{w}x{h}.jpg", gpgm], capture_output=True)
    subprocess.run(["cjpeg", "-quality", "90", "-arithmetic", "-outfile", f"{J}/arith_{w}x{h}.jpg", ppm], capture_output=True)
    subprocess.run(["cjpeg", "-quality", "90", "-rgb", "-outfile", f"{J}/rgbjpeg_{w}x{h}.jpg", ppm], capture_output=True)
    subprocess.run(["cjpeg", "-quality", "90", "-restart", "2", "-outfile", f"{J}/restart_{w}x{h}.jpg", ppm], capture_output=True)
for f in (f"{J}/_t.ppm", f"{J}/_t.pgm"): os.path.exists(f) and os.remove(f)
# white JPEGs and blank scans
for w, h in [(8, 8), (16, 16), (64, 64), (17, 33), (1, 1), (1, 30), (30, 1), (800, 1200)]:
    Image.fromarray(page(w, h)).save(f"{J}/white_{w}x{h}.jpg", quality=95)
    n = page(w, h).astype(np.int16) - rng.integers(0, 6, (h, w, 1)); Image.fromarray(n.clip(0, 255).astype(np.uint8)).save(f"{J}/noisyblank_{w}x{h}.jpg", quality=90)

# ---- WebP ----
W = f"{A}/webp"; os.makedirs(W, exist_ok=True)
for w, h in [(120, 80), (65, 33), (64, 64)]:
    im = Image.fromarray(asym(w, h))
    im.save(f"{W}/lossy_{w}x{h}.webp", quality=90); im.save(f"{W}/ll_{w}x{h}.webp", lossless=True)
    c = np.dstack([asym(w, h), np.where(asym(w, h)[..., 0] == 255, 0, 255).astype(np.uint8)])
    Image.fromarray(c, "RGBA").save(f"{W}/rgba_{w}x{h}.webp", lossless=True, exact=True)
    Image.fromarray(c, "RGBA").save(f"{W}/rgbaq_{w}x{h}.webp", quality=85)

# ---- extreme aspect ratios ----
X = f"{A}/tall"
for w, h in [(1, 16384), (16384, 1), (2, 8192), (8192, 2), (8, 16385), (16385, 8), (1, 4097), (4097, 1), (1, 65535), (65535, 1), (3, 40000), (40000, 3), (16, 65535), (65535, 16)]:
    a = np.full((h, w, 3), 255, np.uint8)
    a[h // 3: max(h // 3 + 1, 2 * h // 3), w // 3: max(w // 3 + 1, 2 * w // 3)] = (10, 90, 200)
    wpng(f"{X}/t_{w}x{h}.png", a)
for w, h in [(8, 16000), (16000, 8), (64, 16000), (16000, 64)]:
    a = page(w, h); a[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = (10, 90, 200)
    Image.fromarray(a).save(f"{X}/tj_{w}x{h}.jpg", quality=92)

# ---- broken / odd files ----
B = f"{A}/broken"; os.makedirs(B, exist_ok=True)
good = open(f"{J}/ex0_420_160x96.jpg", "rb").read(); png = open(f"{T}/bar_mid_200x120.png", "rb").read()
open(f"{B}/empty.jpg", "wb").close()
open(f"{B}/trunc.jpg", "wb").write(good[:len(good) // 2])
open(f"{B}/trunc.png", "wb").write(png[:len(png) // 2])
open(f"{B}/garbage.jpg", "wb").write(bytes(rng.integers(0, 256, 4000, dtype=np.uint8)))
open(f"{B}/garbage.png", "wb").write(b"\x89PNG\r\n\x1a\n" + bytes(rng.integers(0, 256, 4000, dtype=np.uint8)))
open(f"{B}/text.txt", "w").write("not an image\n")
open(f"{B}/headonly.jpg", "wb").write(good[:30])
print("files:", sum(len(f) for _, _, f in os.walk(A)))
