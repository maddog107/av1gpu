#!/usr/bin/env python3
"""random page-like inputs: random sizes, border colours, boxes, alpha, depth, for the fuzzer"""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
from gen_lib import wpng
H = os.path.expanduser("~"); R = f"{H}/adv/rand"; os.makedirs(R, exist_ok=True)
rng = np.random.default_rng(1965)
def size():
    k = rng.integers(0, 4)
    if k == 0: return int(rng.integers(1, 40)), int(rng.integers(1, 40))
    if k == 1: return int(rng.choice([7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])), int(rng.choice([7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]))
    if k == 2: return int(rng.integers(40, 300)), int(rng.integers(40, 300))
    return int(rng.integers(300, 900)), int(rng.integers(100, 900))
for i in range(500):
    w, h = size()
    bg = rng.choice([255, 255, 255, 0, 128, 240]) if rng.random() < .7 else None
    a = np.full((h, w, 3), bg, np.uint8) if bg is not None else np.tile(rng.integers(0, 256, 3, dtype=np.uint8), (h, w, 1))
    if rng.random() < .25:
        a = (a.astype(int) + rng.integers(-4, 5, (h, w, 3))).clip(0, 255).astype(np.uint8)     # paper noise
    for _ in range(int(rng.integers(0, 4))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h)); x1, y1 = int(rng.integers(x0, w)) + 1, int(rng.integers(y0, h)) + 1
        a[y0:y1, x0:x1] = rng.integers(0, 256, 3, dtype=np.uint8)
    k = rng.integers(0, 10)
    if k < 3:
        Image.fromarray(a).save(f"{R}/r{i:03d}.jpg", quality=int(rng.integers(60, 100)), subsampling=int(rng.integers(0, 3)))
    elif k < 5:
        wpng(f"{R}/r{i:03d}.png", a)
    elif k < 7:
        al = np.full((h, w), 255, np.uint8)
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h)); x1, y1 = int(rng.integers(x0, w)) + 1, int(rng.integers(y0, h)) + 1
        al[:] = rng.choice([0, 0, 128, 255]); al[y0:y1, x0:x1] = rng.choice([255, 255, 64])
        wpng(f"{R}/r{i:03d}.png", np.dstack([a, al]))
    elif k < 8:
        wpng(f"{R}/r{i:03d}.png", a[..., 0])
    elif k < 9:
        wpng(f"{R}/r{i:03d}.png", a.astype(np.uint16) * 257 + rng.integers(0, 200, (h, w, 3), dtype=np.uint16).clip(0, 200) * (rng.random() < .5), bits=16)
    else:
        Image.fromarray(a).save(f"{R}/r{i:03d}.webp", lossless=bool(rng.random() < .5))
print(len(os.listdir(R)))
