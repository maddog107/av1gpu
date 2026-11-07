#!/usr/bin/env python3
"""pr65-perf sets (in ~/sets; ~/sets/comics is fetched before this runs):
  thumbs/   512x384 fill and 320-pixel fit JPEGs of the 100 size-range photos, as aws-g6-g7-recon-lazy/scripts/run.sh
  pngwebp/  the size-range photos: odd index as RGB PNG (compress level 6), even index as lossy WebP (q90): the CPU decode route
  st/       cold-start inputs: photo.jpg, big.jpg, d512.jpg, t320.jpg as run.sh picks them; comic.jpg (largest comic page);
            page.png (the first 2886x3950 page as PNG); alpha.png (683-pixel-wide RGBA photo)
  Nothing here depends on either binary."""
import concurrent.futures as cf, glob, os, shutil
import numpy as np
from PIL import Image, ImageOps
H = os.path.expanduser("~")
S = f"{H}/sets"
for d in ("thumbs", "pngwebp", "st"):
    os.makedirs(f"{S}/{d}", exist_ok=True)
fs = sorted(glob.glob(f"{H}/images/*.jpg"))
mp = {}
def one(a):
    i, f = a
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    b = os.path.basename(f)[:-4]
    ImageOps.fit(im, (512, 384), Image.LANCZOS).save(f"{S}/thumbs/d512_{b}.jpg", quality=85)
    t = im.copy(); t.thumbnail((320, 320), Image.LANCZOS); t.save(f"{S}/thumbs/t320_{b}.jpg", quality=85)
    if i % 2:
        im.save(f"{S}/pngwebp/{b}.png", compress_level=6)
    else:
        im.save(f"{S}/pngwebp/{b}.webp", quality=90)
    return f, im.size[0] * im.size[1] / 1e6
with cf.ThreadPoolExecutor(16) as ex:
    mp = dict(ex.map(one, enumerate(fs)))
photo = min((f for f in fs if 2 <= mp[f] <= 3), key=lambda f: abs(mp[f] - 2.5))
big = min((f for f in fs if 10 <= mp[f] <= 12), key=lambda f: abs(mp[f] - 11))
shutil.copy(photo, f"{S}/st/photo.jpg"); shutil.copy(big, f"{S}/st/big.jpg")
b = os.path.basename(photo)[:-4]
shutil.copy(f"{S}/thumbs/d512_{b}.jpg", f"{S}/st/d512.jpg"); shutil.copy(f"{S}/thumbs/t320_{b}.jpg", f"{S}/st/t320.jpg")
cs = sorted(glob.glob(f"{S}/comics/*.jpg"), key=lambda f: -Image.open(f).size[0] * Image.open(f).size[1])
shutil.copy(cs[0], f"{S}/st/comic.jpg")
pg = next(f for f in cs if Image.open(f).size == (2886, 3950))
Image.open(pg).convert("RGB").save(f"{S}/st/page.png", compress_level=6)
a = ImageOps.exif_transpose(Image.open(photo)).convert("RGB"); a.thumbnail((1024, 1024), Image.LANCZOS)
w, h = a.size; y, x = np.mgrid[0:h, 0:w]
al = np.clip(255 * (1.3 - np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2))), 0, 255).astype(np.uint8)
a.putalpha(Image.fromarray(al)); a.save(f"{S}/st/alpha.png")
for d in ("comics", "thumbs", "pngwebp", "st"):
    ff = sorted(glob.glob(f"{S}/{d}/*"))
    print(d, len(ff), "files", round(sum(Image.open(f).size[0] * Image.open(f).size[1] for f in ff) / 1e6, 1), "MP")
