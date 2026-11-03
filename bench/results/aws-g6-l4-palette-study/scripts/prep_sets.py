#!/usr/bin/env python3
"""Fetch and build the palette study's image sets (on the instance).

  prep_sets.py MANIFEST.csv GAMUT_MANIFEST.csv SRC_TREE RENDER_DIR OUT

Downloads every manifest row (SHA-256 checked), then builds OUT/sets/<set>/:
  comics    the scanned comic pages, JPEG as served
  flat_png  flat-colour art as 8-bit RGB PNG: transparency composited onto white, ICC / text chunks dropped; SVG
            originals rendered with rsvg-convert (long side 2560 px, white background)
  flat_jpg  flat_png saved by Pillow as JPEG quality 90, 4:2:0 (the forum-JPEG version of the same art)
  painted   Pepper&Carrot pages, JPEG as published
  screen    the 23 graphics of the lossless comparison (mk_graphics.py), the Commons screenshots (RGB, as flat_png)
            and the headless-browser renders (RENDER_DIR)
  photo     the 24 Kodak PNGs and 20 size-range JPEGs (bench/openimages_v4_gamut.csv: the first 4 of each size class)
and OUT/sets.json (set, file, width, height, SHA-256 of each file the encoders read)."""
import concurrent.futures as cf
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from PIL import Image

UA = "av1gpu-bench/0.1 (https://github.com/maddog107/av1gpu; palette study)"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fetch(row, d):
    path = os.path.join(d, row["file"])
    if not (os.path.exists(path) and sha256(path) == row["sha256"]):
        for t in range(6):   # Wikimedia answers 429 to bursts: back off
            try:
                data = urllib.request.urlopen(urllib.request.Request(row["url"], headers={"User-Agent": UA}), timeout=300).read()
                break
            except Exception:
                if t == 5:
                    raise
                time.sleep(15 * (t + 1))
        open(path + ".part", "wb").write(data)
        if sha256(path + ".part") != row["sha256"]:
            raise RuntimeError(f"{row['file']}: sha256 mismatch")
        os.replace(path + ".part", path)
    return path


def to_rgb_png(src, dst):
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im.convert("RGB").save(dst, compress_level=6)


def render_svg(src, dst, w, h):
    tmp = dst + ".tmp.png"
    side = ["-w", "2560"] if w >= h else ["-h", "2560"]
    subprocess.run(["rsvg-convert", "-a"] + side + ["-b", "white", "-o", tmp, src], check=True)
    to_rgb_png(tmp, dst)
    os.remove(tmp)


def main():
    man, gamut, src_tree, render_dir, out = sys.argv[1:6]
    orig = os.path.join(out, "orig")
    sets = os.path.join(out, "sets")
    for d in [orig] + [os.path.join(sets, s) for s in ("comics", "flat_png", "flat_jpg", "painted", "screen", "photo")]:
        os.makedirs(d, exist_ok=True)
    rows = list(csv.DictReader(open(man, newline="")))
    with cf.ThreadPoolExecutor(3) as ex:
        paths = list(ex.map(lambda r: fetch(r, orig), rows))
    print(len(paths), "downloaded / verified", flush=True)
    for r, p in zip(rows, paths):
        cls, f = r["cls"], r["file"]
        if cls in ("comics", "painted"):
            shutil.copy(p, os.path.join(sets, cls, f))
        elif cls == "flat":
            dst = os.path.join(sets, "flat_png", os.path.splitext(f)[0] + ".png")
            if f.endswith(".svg"):
                render_svg(p, dst, int(r["width"]), int(r["height"]))
            else:
                to_rgb_png(p, dst)
            Image.open(dst).save(os.path.join(sets, "flat_jpg", os.path.splitext(f)[0] + ".jpg"), quality=90, subsampling=2)
        elif cls == "screen":
            to_rgb_png(p, os.path.join(sets, "screen", f))
        elif cls == "photo":
            shutil.copy(p, os.path.join(sets, "photo", f))
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_graphics.py"), src_tree,
                    os.path.join(out, "graphics")], check=True)
    for g in sorted(glob.glob(os.path.join(out, "graphics", "*.png"))):
        to_rgb_png(g, os.path.join(sets, "screen", "screen_synth_" + os.path.basename(g)))
    for g in sorted(glob.glob(os.path.join(render_dir, "*.png"))):
        to_rgb_png(g, os.path.join(sets, "screen", os.path.basename(g)))
    # 20 size-range photos: the first 4 of each size class of the gamut manifest
    grows = list(csv.DictReader(open(gamut, newline="")))
    pick, seen = [], {}
    for r in grows:
        if seen.get(r["size_class"], 0) < 4:
            seen[r["size_class"]] = seen.get(r["size_class"], 0) + 1
            pick.append(r)
    with cf.ThreadPoolExecutor(8) as ex:
        gp = list(ex.map(lambda r: fetch(dict(r, file=r["file"]), orig), pick))
    for r, p in zip(pick, gp):
        shutil.copy(p, os.path.join(sets, "photo", "photo_" + r["file"]))
    info = []
    for s in sorted(os.listdir(sets)):
        for f in sorted(os.listdir(os.path.join(sets, s))):
            p = os.path.join(sets, s, f)
            im = Image.open(p)
            info.append(dict(set=s, file=f, width=im.width, height=im.height, sha256=sha256(p)))
    json.dump(info, open(os.path.join(out, "sets.json"), "w"), indent=1)
    for s in sorted({i["set"] for i in info}):
        sel = [i for i in info if i["set"] == s]
        print(s, len(sel), "images", round(sum(i["width"] * i["height"] for i in sel) / 1e6, 1), "MP")


if __name__ == "__main__":
    main()
