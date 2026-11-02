#!/usr/bin/env python3
"""The 23 graphics of docs/RESULTS.md's lossless comparison, rebuilt from the source tree (the graphics part of
bench/results/aws-g6-l4-speed-v06/scripts/mk_ll_sets.py, unchanged apart from the paths):
  16 of docs/charts/*.svg rendered with rsvg-convert at 2x (the 16 charts that existed then: all but formats-* and
  metrics-*) and 7 synthetic 1920x1080 screenshots (text page, dark text page, terminal, code editor, UI, chart, table).
  mk_graphics.py SRC_TREE OUTDIR"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, R = sys.argv[1], sys.argv[2]
os.makedirs(R, exist_ok=True)
for s in sorted(glob.glob(f"{SRC}/docs/charts/*.svg")):
    if os.path.basename(s).startswith(("formats-", "metrics-")):
        continue
    o = f"{R}/chart_{os.path.basename(s)[:-4]}.png"
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
src_lines = open(f"{SRC}/av1core/src/encoder.cpp").read().splitlines()


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


text_page(f"{R}/shot_text.png")
text_page(f"{R}/shot_text_dark.png", dark=True)
terminal(f"{R}/shot_terminal.png")
editor(f"{R}/shot_editor.png")
ui(f"{R}/shot_ui.png")
chart(f"{R}/shot_chart.png")
table(f"{R}/shot_table.png")
print(len(glob.glob(f"{R}/*.png")), "graphics in", R)
