#!/usr/bin/env python3
"""Lossless test sets (run on the instance; outputs under ~/ll):
  kodak/      the 24 Kodak Lossless True Color Image Suite PNGs (768x512 photos, r0k.us), RGB 8-bit
  graphics/   screenshots / graphics: the repository's 16 SVG charts rendered with rsvg-convert (2x), 6 synthetic
              1920x1080 screenshots (text page, terminal, UI with flat panels and gradients, code editor, chart, table)
  cmp/        kodak + graphics: the comparison table set
  png16_10/   16-bit RGB PNGs holding 10 significant bits (libavif expansion of 10-bit values: 2x2 float means of Kodak
              images, plus two smooth 10-bit gradients); png16_8/: 16-bit PNGs with 8 significant bits (v * 257)
  png16_full/ genuine 16-bit PNGs (must be rejected)
  gray/       8-bit gray PNGs and 16-bit gray PNGs with 10 significant bits (monochrome)
  misc/       palette PNGs, RGBA PNGs (alpha not coded), odd sizes (1x1, 7x13, 65x65, 129x1, 4100x40 = forced tiles,
              3x5000), a large photo PNG (Kodak 4x tiled + noise, 6144x4096 = 25 MP)
  webp/       lossless and lossy WebP (cwebp) from Kodak
  jpeg/       bench JPEGs (4:2:0), 4:4:4 and gray JPEGs (PIL), odd sizes, a 4:2:2 JPEG (must be rejected)"""
import glob
import os
import struct
import subprocess
import urllib.request
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = os.path.expanduser("~")
R = f"{H}/ll"
for d in ["kodak", "graphics", "cmp", "png16_10", "png16_8", "png16_full", "gray", "misc", "webp", "jpeg"]:
    os.makedirs(f"{R}/{d}", exist_ok=True)


def write_png(path, a, bits):
    """a: HxW (gray) or HxWx3 (RGB) integer array -> PNG with the given bit depth (filter 0 rows)"""
    a = np.asarray(a)
    h, w = a.shape[:2]
    ct = 0 if a.ndim == 2 else 2
    dt = ">u2" if bits == 16 else "u1"
    raw = b"".join(b"\x00" + a[y].astype(dt).tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ct, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


def expand10(v):   # libavif's 10 -> 16-bit expansion, float32 as in reformat.c
    f = (v.astype(np.float32) - np.float32(0)) / np.float32(1023)
    return (np.float32(0.5) + f * np.float32(65535)).astype(np.uint16)


# ---- Kodak ----
for i in range(1, 25):
    p = f"{R}/kodak/kodim{i:02d}.png"
    if not os.path.exists(p):
        try:
            urllib.request.urlretrieve(f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", p)
        except Exception as e:
            print("kodak fetch failed", i, e)
kodak = sorted(glob.glob(f"{R}/kodak/*.png"))
print("kodak", len(kodak))
# ---- graphics ----
for s in sorted(glob.glob(f"{H}/br/docs/charts/*.svg")):
    o = f"{R}/graphics/chart_{os.path.basename(s)[:-4]}.png"
    subprocess.run(["rsvg-convert", "-z", "2", "-b", "white" if "light" in s else "#0d1117", "-o", o, s], check=False)
fonts = sorted(glob.glob("/usr/share/fonts/truetype/dejavu/*.ttf"))
mono = next((f for f in fonts if "SansMono.ttf" in f), None)
sans = next((f for f in fonts if f.endswith("DejaVuSans.ttf")), None)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


rng = np.random.default_rng(7)
words = ("lossless encoding keeps every sample exactly the walsh hadamard transform of the residual is coded with "
         "the same entropy coder as lossy streams while prediction comes from the neighbouring samples").split()
src_lines = open(f"{H}/br/av1core/src/encoder.cpp").read().splitlines()


def text_page(path, dark=False):
    im = Image.new("RGB", (1920, 1080), (30, 30, 30) if dark else (255, 255, 255))
    d = ImageDraw.Draw(im)
    f1, f2 = font(sans, 34), font(sans, 18)
    d.text((80, 40), "Lossless AVIF on the GPU", font=f1, fill=(230, 230, 230) if dark else (20, 20, 20))
    y = 110
    while y < 1040:
        line = " ".join(rng.choice(words, 14))
        d.text((80, y), line, font=f2, fill=(200, 200, 200) if dark else (40, 40, 40))
        y += 26
    im.save(path)


def terminal(path):
    im = Image.new("RGB", (1920, 1080), (12, 12, 12))
    d = ImageDraw.Draw(im)
    f = font(mono, 16)
    y = 8
    for k, l in enumerate(src_lines[400:470]):
        d.text((10, y), l[:200], font=f, fill=(0, 255, 120) if k % 7 else (255, 210, 0))
        y += 15
    im.save(path)


def editor(path):
    im = Image.new("RGB", (1920, 1080), (40, 42, 54))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 260, 1080], fill=(33, 34, 44))
    d.rectangle([0, 1050, 1920, 1080], fill=(98, 114, 164))
    f = font(mono, 15)
    cols = [(248, 248, 242), (139, 233, 253), (80, 250, 123), (255, 121, 198), (241, 250, 140), (189, 147, 249)]
    for k in range(60):
        d.text((10, 10 + 17 * k), f"file_{k:02d}.cpp", font=f, fill=(150, 150, 170))
    y = 10
    for k, l in enumerate(src_lines[600:660]):
        d.text((280, y), f"{600 + k:5d}  " + l[:180], font=f, fill=cols[k % len(cols)])
        y += 17
    im.save(path)


def ui(path):
    im = Image.new("RGB", (1920, 1080), (245, 246, 250))
    a = np.asarray(im).copy()
    a[:80] = (np.linspace(40, 90, 1920)[None, :, None] * np.array([1.0, 1.4, 2.4])[None, None, :]).clip(0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = font(sans, 20)
    for k in range(12):
        x, y = 60 + (k % 4) * 460, 140 + (k // 4) * 300
        d.rounded_rectangle([x, y, x + 420, y + 260], 14, fill=(255, 255, 255), outline=(220, 222, 230), width=2)
        d.text((x + 20, y + 20), f"Card {k + 1}", font=f, fill=(30, 30, 40))
        d.rounded_rectangle([x + 20, y + 200, x + 160, y + 240], 8, fill=(66 + 10 * k, 120, 220))
        d.text((x + 40, y + 208), "Open", font=f, fill=(255, 255, 255))
        for j in range(5):
            d.text((x + 20, y + 60 + 25 * j), " ".join(rng.choice(words, 5)), font=font(sans, 16), fill=(90, 90, 100))
    im.save(path)


def chart(path):
    im = Image.new("RGB", (1920, 1080), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for k in range(11):
        d.line([(120, 100 + 80 * k), (1860, 100 + 80 * k)], fill=(225, 225, 225), width=1)
    pal = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    for s in range(4):
        pts = [(120 + 29 * i, 900 - 60 * s - 300 * abs(np.sin(i / (9.0 + 3 * s)))) for i in range(61)]
        d.line(pts, fill=pal[s], width=4)
        for p in pts[::6]:
            d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=pal[s])
    d.text((120, 30), "throughput vs quality", font=font(sans, 30), fill=(20, 20, 20))
    im.save(path)


def table(path):
    im = Image.new("RGB", (1920, 1080), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = font(sans, 17)
    for r in range(38):
        if r % 2:
            d.rectangle([40, 30 + 27 * r, 1880, 57 + 27 * r], fill=(242, 245, 250))
        for c in range(8):
            v = rng.integers(0, 100000) if r else None
            d.text((60 + 230 * c, 34 + 27 * r), f"column {c}" if r == 0 else f"{v:,}", font=f, fill=(20, 20, 20))
    im.save(path)


text_page(f"{R}/graphics/shot_text.png")
text_page(f"{R}/graphics/shot_text_dark.png", dark=True)
terminal(f"{R}/graphics/shot_terminal.png")
editor(f"{R}/graphics/shot_editor.png")
ui(f"{R}/graphics/shot_ui.png")
chart(f"{R}/graphics/shot_chart.png")
table(f"{R}/graphics/shot_table.png")
for p in glob.glob(f"{R}/graphics/*.png") + kodak:
    dst = f"{R}/cmp/{os.path.basename(p)}"
    if not os.path.exists(dst):
        os.symlink(p, dst)
# ---- 16-bit sets ----
for i, p in enumerate(kodak[:6]):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
    b = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2].reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    v10 = np.clip(np.rint(b * 1023.0 / 255.0), 0, 1023).astype(np.int32)
    write_png(f"{R}/png16_10/k{i}_10in16.png", expand10(v10), 16)
    if i < 3:
        write_png(f"{R}/png16_8/k{i}_8in16.png", a.astype(np.uint16) * 257, 16)
    if i < 2:
        write_png(f"{R}/png16_full/k{i}_full16.png", np.clip(np.rint(b * 257.0) + rng.integers(0, 64, b.shape), 0, 65535).astype(np.uint16), 16)
yy, xx = np.mgrid[0:600, 0:1001].astype(np.float64)
g = np.stack([200 + 300 * xx / 1000, 400 + 100 * yy / 599, 800 - 200 * (xx + yy) / 1600], -1)
write_png(f"{R}/png16_10/grad10_a.png", expand10(np.rint(g).astype(np.int32)), 16)
g2 = np.stack([512 + 20 * np.sin(xx / 37.0) * np.cos(yy / 23.0)] * 3, -1)
write_png(f"{R}/png16_10/grad10_b.png", expand10(np.rint(g2).astype(np.int32)), 16)
# ---- gray ----
for i, p in enumerate(kodak[6:9]):
    Image.open(p).convert("L").save(f"{R}/gray/g{i}.png")
for i, p in enumerate(kodak[9:11]):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float64) * 1023.0 / 255.0
    write_png(f"{R}/gray/g10_{i}.png", expand10(np.rint(a).astype(np.int32)), 16)
# ---- misc ----
for i, p in enumerate(kodak[11:14]):
    Image.open(p).convert("RGB").quantize(64 + 64 * i).save(f"{R}/misc/pal{i}.png")
for i, p in enumerate(kodak[14:16]):
    im = Image.open(p).convert("RGBA")
    a = np.asarray(im).copy()
    a[..., 3] = (np.arange(a.shape[1]) % 256)[None, :]
    Image.fromarray(a).save(f"{R}/misc/rgba{i}.png")
k0 = np.asarray(Image.open(kodak[16]).convert("RGB"))
for (w, h) in [(1, 1), (7, 13), (65, 65), (129, 1), (3, 5000 // 8), (767, 511)]:
    Image.fromarray(np.ascontiguousarray(np.resize(k0, (h, w, 3)))).save(f"{R}/misc/odd_{w}x{h}.png")
wide = np.tile(k0, (1, 6, 1))[:40, :4100]
Image.fromarray(np.ascontiguousarray(wide)).save(f"{R}/misc/wide_4100x40.png")
big = np.tile(np.asarray(Image.open(kodak[17]).convert("RGB")).astype(np.int32), (8, 8, 1))[:4096, :6144]
big = np.clip(big + rng.integers(-2, 3, big.shape), 0, 255).astype(np.uint8)
Image.fromarray(big).save(f"{R}/misc/big_6144x4096.png", compress_level=1)
# ---- WebP ----
for i, p in enumerate(kodak[18:22]):
    if i < 2:
        subprocess.run(["cwebp", "-quiet", "-lossless", p, "-o", f"{R}/webp/kl{i}.webp"], check=False)
    else:
        subprocess.run(["cwebp", "-quiet", "-q", "80", p, "-o", f"{R}/webp/ky{i}.webp"], check=False)
# ---- JPEG ----
bench = sorted(glob.glob(f"{H}/images/*.jpg"))
for i, p in enumerate(bench[:12]):
    os.symlink(p, f"{R}/jpeg/b{i}_{os.path.basename(p)}") if not os.path.exists(f"{R}/jpeg/b{i}_{os.path.basename(p)}") else None
for i, p in enumerate(kodak[:4]):
    im = Image.open(p).convert("RGB")
    im.save(f"{R}/jpeg/k{i}_444.jpg", quality=92, subsampling=0)
    im.save(f"{R}/jpeg/k{i}_420.jpg", quality=85, subsampling=2)
    if i < 2:
        im.convert("L").save(f"{R}/jpeg/k{i}_gray.jpg", quality=90)
    if i == 0:
        im.save(f"{R}/jpeg/k{i}_422.jpg", quality=90, subsampling=1)
        im.crop((0, 0, 333, 257)).save(f"{R}/jpeg/k{i}_odd_420.jpg", quality=88, subsampling=2)
        im.crop((0, 0, 101, 77)).save(f"{R}/jpeg/k{i}_odd_444.jpg", quality=88, subsampling=0)
for d in sorted(os.listdir(R)):
    print(d, len(os.listdir(f"{R}/{d}")))
