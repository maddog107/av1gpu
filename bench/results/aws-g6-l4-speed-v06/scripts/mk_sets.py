#!/usr/bin/env python3
"""Identity / speed sets for the v0.6.0 release check (run on the instance after mk_ll_sets.py; outputs in ~/sets):
  thumbs/  small JPEGs from the 100 size-range photos, made with ImageMagick: d512_* = 512x384 fill-and-crop
           downscales (q85), t320_* = fit-in-320x320 thumbnails (q80), c512_* = 512x384 crops (q90) of the photos
           of >= 4 MP (centre and upper-left quarter point)
  odd/     odd sizes as RGB PNG, 4:2:0 JPEG and 4:4:4 JPEG (Kodak 17 resampled): 1x1 2x2 3x5 5x3 7x7 63x65 65x63
           127x129 4097x17 17x4097
  gray/    ll/gray (8-bit, and 16-bit with 10 significant bits) plus gray + alpha PNGs: 8-bit LA (gradient, binary
           mask, all-opaque alpha) and a 16-bit LA
  webp/    ll/webp (2 lossless, 2 lossy from Kodak) plus RGBA WebPs from the RGBA set (2 lossy, 1 lossless -exact)
  single/  10 files for single-file mode (JPEGs from 0.2 to 24 MP, Kodak PNGs, RGBA, gray, odd)
  mixed.txt  a --batch LIST of JPEGs, PNGs (RGB, RGBA, gray, odd) and WebPs"""
import concurrent.futures as cf, glob, os, struct, subprocess, zlib
import numpy as np
from PIL import Image

H = os.path.expanduser("~")
S = f"{H}/sets"
for d in ["thumbs", "odd", "gray", "webp", "single"]:
    os.makedirs(f"{S}/{d}", exist_ok=True)
photos = sorted(glob.glob(f"{H}/images/*.jpg"))
kodak = sorted(glob.glob(f"{H}/ll/kodak/*.png"))
rgba = sorted(glob.glob(f"{H}/data/rgba/set/*.png"))

# ---- thumbnails (ImageMagick) ----
jobs = []
for f in photos:
    b = os.path.splitext(os.path.basename(f))[0]
    jobs.append(["convert", f, "-resize", "512x384^", "-gravity", "center", "-extent", "512x384", "-quality", "85",
                 f"{S}/thumbs/d512_{b}.jpg"])
    jobs.append(["convert", f, "-thumbnail", "320x320", "-quality", "80", f"{S}/thumbs/t320_{b}.jpg"])
    w, h = Image.open(f).size
    if w * h >= 4e6:
        for k, (x, y) in enumerate([((w - 512) // 2, (h - 384) // 2), (w // 4, h // 4)]):
            jobs.append(["convert", f, "-crop", f"512x384+{x}+{y}", "+repage", "-quality", "90", f"{S}/thumbs/c512_{k}_{b}.jpg"])
with cf.ThreadPoolExecutor(16) as ex:
    rcs = list(ex.map(lambda c: subprocess.run(c, capture_output=True).returncode, jobs))
print("thumbs", len(jobs), "convert jobs, failures", sum(1 for r in rcs if r))


def write_png(path, a, bits, ct):
    a = np.asarray(a)
    h, w = a.shape[:2]
    dt = ">u2" if bits == 16 else "u1"
    raw = b"".join(b"\x00" + a[y].astype(dt).tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ct, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---- odd sizes ----
k17 = Image.open(kodak[16]).convert("RGB")
for (w, h) in [(1, 1), (2, 2), (3, 5), (5, 3), (7, 7), (63, 65), (65, 63), (127, 129), (4097, 17), (17, 4097)]:
    im = k17.resize((w, h), Image.LANCZOS)
    im.save(f"{S}/odd/o{w}x{h}.png")
    im.save(f"{S}/odd/o{w}x{h}_420.jpg", quality=90, subsampling=2)
    im.save(f"{S}/odd/o{w}x{h}_444.jpg", quality=90, subsampling=0)

# ---- gray and gray + alpha ----
for f in sorted(glob.glob(f"{H}/ll/gray/*.png")):
    dst = f"{S}/gray/{os.path.basename(f)}"
    if not os.path.exists(dst):
        os.symlink(f, dst)
for i, p in enumerate(kodak[19:22]):
    g = np.asarray(Image.open(p).convert("L"))
    hh, ww = g.shape
    if i == 0:
        a = np.tile((np.arange(ww) * 255 // max(ww - 1, 1)).astype(np.uint8), (hh, 1))
    elif i == 1:
        yy, xx = np.mgrid[0:hh, 0:ww]
        a = np.where((xx - ww / 2) ** 2 + (yy - hh / 2) ** 2 < (min(ww, hh) / 3) ** 2, 255, 0).astype(np.uint8)
    else:
        a = np.full((hh, ww), 255, np.uint8)
    Image.fromarray(np.stack([g, a], -1), "LA").save(f"{S}/gray/la{i}.png")
g16 = np.asarray(Image.open(kodak[22]).convert("L"), dtype=np.uint16) * 257
a16 = np.tile((np.arange(g16.shape[1]) * 65535 // (g16.shape[1] - 1)).astype(np.uint16), (g16.shape[0], 1))
write_png(f"{S}/gray/la16.png", np.stack([g16, a16], -1), 16, 4)

# ---- WebP ----
for f in sorted(glob.glob(f"{H}/ll/webp/*.webp")):
    dst = f"{S}/webp/{os.path.basename(f)}"
    if not os.path.exists(dst):
        os.symlink(f, dst)
ra = [p for p in rgba if Image.open(p).mode in ("RGBA", "LA", "P")][:3] or rgba[:3]
for i, p in enumerate(ra):
    o = f"{S}/webp/rgba{i}.webp"
    if i < 2:
        subprocess.run(["cwebp", "-quiet", "-q", "80", "-alpha_q", "90", p, "-o", o], check=False)
    else:
        subprocess.run(["cwebp", "-quiet", "-lossless", "-exact", p, "-o", o], check=False)

# ---- single-file mode ----
by_mp = sorted(photos, key=lambda f: Image.open(f).size[0] * Image.open(f).size[1])
pick = [by_mp[i] for i in (0, 20, 50, 80, len(by_mp) - 1)] + kodak[:2] + rgba[:1] + [f"{S}/gray/g0.png", f"{S}/odd/o63x65_420.jpg"]
for f in pick:
    dst = f"{S}/single/{os.path.basename(f)}"
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(f), dst)

# ---- mixed list ----
mixed = photos[::10] + kodak[::4] + rgba[::8] + sorted(glob.glob(f"{S}/gray/*.png"))[:3] + \
    sorted(glob.glob(f"{S}/webp/*.webp")) + [f"{S}/odd/o3x5.png", f"{S}/odd/o127x129_444.jpg"]
open(f"{S}/mixed.txt", "w").write("".join(m + "\n" for m in mixed))
for d in sorted(os.listdir(S)):
    p = f"{S}/{d}"
    print(d, len(os.listdir(p)) if os.path.isdir(p) else len(open(p).read().splitlines()))
