#!/usr/bin/env python3
"""~/sets/border: SYNTHETIC scan borders.  Each page of ~/sets/comics pasted on a paper-coloured canvas, margins 3..7 % per side
(seeded by the file name), saved as JPEG q92 4:2:0.  Odd pages: a flat margin (a digital border, as --trim --fuzz 0% can find).
Even pages: paper noise (sigma 1.5 steps), so only a --fuzz finds it.  The page content is untouched.  This is what a scanned
album looks like to the trim path: most of the image kept, a ring of paper removed."""
import concurrent.futures as cf, glob, hashlib, os
import numpy as np
from PIL import Image
H = os.path.expanduser("~"); D = f"{H}/sets/border"; os.makedirs(D, exist_ok=True)
fs = sorted(glob.glob(f"{H}/sets/comics/*.jpg"))
def one(a):
    i, f = a
    r = np.random.default_rng(int(hashlib.sha256(os.path.basename(f).encode()).hexdigest()[:8], 16))
    p = np.asarray(Image.open(f).convert("RGB")); h, w = p.shape[:2]
    l, rr = (r.uniform(.03, .07, 2) * w).astype(int); t, b = (r.uniform(.03, .07, 2) * h).astype(int)
    H_, W_ = h + t + b, w + l + rr
    paper = np.array([248, 246, 240], float)
    c = np.tile(paper, (H_, W_, 1))
    if i % 2 == 0: c += r.normal(0, 1.5, c.shape)
    c = np.clip(c, 0, 255).astype(np.uint8); c[t:t + h, l:l + w] = p
    Image.fromarray(c).save(f"{D}/{os.path.basename(f)}", quality=92, subsampling=2)
    return w * h, W_ * H_
with cf.ThreadPoolExecutor(16) as ex:
    v = list(ex.map(one, enumerate(fs)))
print(len(v), "pages", round(sum(a for a, _ in v) / 1e6, 1), "MP of page,", round(sum(b for _, b in v) / 1e6, 1), "MP with margins:", round(100 * (1 - sum(a for a, _ in v) / sum(b for _, b in v)), 1), "% border")
