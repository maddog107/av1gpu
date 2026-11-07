#!/usr/bin/env python3
"""~/sets/fuzz: images aimed at the places where an early exit in the --trim measurement could change the box.
Every image has a seed-derived case: a flat page with 1-3 specks in rows and columns the coarse pass misses (y % 64 >= 4), on the
edges, or next to corners; four different corners; the bottom-right rule (bottom-left colour along the bottom edge, a differing
bottom-right corner, with pixels left of the right edge that differ only from it); all border; thin edge lines; gradients;
near-threshold noise.  Sizes 1..300, some long thin ones and a few pages.  Formats: RGB PNG, gray PNG, RGBA PNG (alpha cases),
16-bit RGB PNG (10-bit planes), JPEG 4:2:0 / 4:4:4 / gray.  mk_trimfuzz.py SEED N"""
import os, random, sys
import numpy as np
from PIL import Image
seed, N = int(sys.argv[1]), int(sys.argv[2])
H = os.path.expanduser("~"); D = f"{H}/sets/fuzz"; os.makedirs(D, exist_ok=True)
for f in os.listdir(D): os.remove(f"{D}/{f}")
R = random.Random(seed); NP = np.random.default_rng(seed)
def col(r):
    return [r.randrange(256) for _ in range(3)] if r.random() < .5 else r.choice([[255] * 3, [0] * 3, [250, 250, 245], [128] * 3])
def size(r):
    c = r.random()
    if c < .55: return r.randrange(1, 301), r.randrange(1, 301)
    if c < .75: return r.randrange(60, 900), r.randrange(60, 900)
    if c < .85: return r.choice([(4097, 17), (17, 4097), (2000, 3), (3, 2000), (513, 129), (129, 513)])
    if c < .93: return r.randrange(1000, 2000), r.randrange(1000, 2600)
    return r.choice([(64, 64), (128, 64), (63, 65), (65, 63), (256, 256), (257, 255)])
def speck(a, r, c0):
    h, w = a.shape[:2]
    y = r.randrange(h)
    if h > 8 and r.random() < .7:
        for _ in range(20):
            y = r.randrange(h)
            if y % 64 >= 4: break
    x = r.choice([0, w - 1, r.randrange(w), min(w - 1, 1), max(0, w - 2)])
    v = col(r)
    if r.random() < .5:
        d = r.choice([1, 2, 5, 12, 26, 27, 40])
        v = [min(255, max(0, c + r.choice([-d, d]))) for c in c0]
    a[y, x] = v
def page(i):
    r = R; w, h = size(r); k = r.randrange(8)
    c0 = col(r); a = np.zeros((h, w, 3), np.uint8); a[:] = c0
    if k == 0:                        # specks
        for _ in range(r.randrange(1, 4)): speck(a, r, c0)
    elif k == 1:                      # four corners
        a[0, 0] = col(r); a[0, -1] = col(r); a[-1, 0] = col(r); a[-1, -1] = col(r)
        for _ in range(r.randrange(0, 3)): speck(a, r, c0)
    elif k == 2:                      # the bottom-right rule
        cb = col(r); e = r.randrange(1, min(h, 40) + 1); a[-e:, :] = cb
        a[-1, -1] = col(r)
        if w > 2 and e > 1: a[-r.randrange(1, e + 1), r.randrange(w - 1)] = col(r)
    elif k == 3:                      # all border
        pass
    elif k == 4:                      # thin lines on or near the edges, and a box of content
        y0 = r.randrange(h); x0 = r.randrange(w); a[y0:y0 + r.randrange(1, 40), x0:x0 + r.randrange(1, 60)] = col(r)
        if r.random() < .5: a[r.choice([0, h - 1]), :] = col(r)
        if r.random() < .5: a[:, r.choice([0, w - 1])] = col(r)
    elif k == 5:                      # gradient
        xx = np.linspace(0, 255, w)[None, :]; yy = np.linspace(0, 255, h)[:, None]
        a[..., 0] = xx.astype(np.uint8); a[..., 1] = yy.astype(np.uint8); a[..., 2] = ((xx + yy) / 2).astype(np.uint8)
    elif k == 6:                      # noise around the corner colour, near common --fuzz thresholds
        n = NP.integers(-r.choice([1, 6, 20, 30]), r.choice([2, 7, 21, 31]), a.shape)
        a = np.clip(a.astype(int) + n, 0, 255).astype(np.uint8)
    else:                             # a margin of k pixels around random content
        m = r.randrange(0, min(w, h) // 2 + 1)
        if h > 2 * m and w > 2 * m: a[m:h - m, m:w - m] = NP.integers(0, 256, (h - 2 * m, w - 2 * m, 3), dtype=np.uint8)
    im = Image.fromarray(a); f = r.randrange(7); b = f"{D}/f{i:04d}"
    if f == 0: im.save(b + ".png", compress_level=1)
    elif f == 1: im.convert("L").save(b + ".png", compress_level=1)
    elif f == 2:
        al = np.full((h, w), 255, np.uint8)
        c = r.randrange(3)
        if c == 0: al[r.randrange(h), r.randrange(w)] = r.choice([0, 128, 254])
        elif c == 1: al[:] = 0; al[h // 4:3 * h // 4 + 1, w // 4:3 * w // 4 + 1] = 255
        else: al[:] = np.linspace(0, 255, w).astype(np.uint8)
        Image.fromarray(np.dstack([a, al]), "RGBA").save(b + ".png", compress_level=1)
    elif f == 3:                      # 16-bit RGB PNG: written by hand (Pillow has no 48-bit RGB writer)
        import struct, zlib
        d = (a.astype(np.uint16) * 257).astype(">u2").tobytes(); st = w * 6
        raw = b"".join(b"\0" + d[y * st:(y + 1) * st] for y in range(h))
        ch = lambda t, x: struct.pack(">I", len(x)) + t + x + struct.pack(">I", zlib.crc32(t + x) & 0xFFFFFFFF)
        open(b + ".png", "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 1)) + ch(b"IEND", b""))
    elif f == 4: im.save(b + ".jpg", quality=95, subsampling=2)
    elif f == 5: im.save(b + ".jpg", quality=95, subsampling=0)
    else: im.convert("L").save(b + ".jpg", quality=95)
for i in range(N): page(i)
print(len(os.listdir(D)), "files")
