#!/usr/bin/env python3
"""Test sets under ~/data (instance); each list repeats its images (uniquely named symlinks) to a minimum MP.

jpeg.txt  the size-range JPEGs (99-100 photos), >= 1,400 MP
png.txt   the same photos as RGB PNGs (Pillow defaults), >= 950 MP
mix.txt   per size class 5 PNG + 3 WebP (q90, one lossless) + 12 JPEG (as tbf.py's mix), >= 950 MP
fewpng.txt / fewjpg.txt  3 small images each (hang test)
"""
import csv, glob, os, random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

H = os.path.expanduser("~")
SRC, D = f"{H}/images_gamut", f"{H}/data"
MAN = f"{H}/av1gpu/bench/openimages_v4_gamut.csv"


def to_png(a):
    src, dst = a
    if not os.path.exists(dst):
        im = Image.open(src)
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        im.save(dst)
    return dst


def to_webp(a):
    src, dst, ll = a
    if not os.path.exists(dst):
        Image.open(src).convert("RGB").save(dst, "WEBP", quality=90, lossless=ll, method=4)
    return dst


def mp(p):
    w, h = Image.open(p).size
    return w * h / 1e6


def rep(name, files, min_mp):
    d = f"{D}/{name}"
    os.makedirs(d, exist_ok=True)
    sizes = [mp(f) for f in files]
    tot, out, k = 0.0, [], 0
    while tot < min_mp:
        for f, s in zip(files, sizes):
            l = f"{d}/r{k}_{os.path.basename(f)}"
            if not os.path.lexists(l):
                os.symlink(f, l)
            out.append(l)
            tot += s
        k += 1
    open(f"{D}/{name}.txt", "w").write("".join(p + "\n" for p in out))
    print(name, len(files), "files", len(out), "entries", round(tot, 1), "MP")


def main():
    rows = [r for r in csv.DictReader(open(MAN)) if os.path.exists(os.path.join(SRC, r["file"]))]
    for r in rows:
        r["path"] = os.path.join(SRC, r["file"])
    os.makedirs(f"{D}/src", exist_ok=True)
    jpgs = sorted(r["path"] for r in rows)
    pj = [(p, f"{D}/src/{os.path.basename(p)[:-4]}.png") for p in jpgs]
    rnd = random.Random(15)
    mix, wj = [], []
    for cls in ("tiny", "small", "typical", "large", "huge"):
        rs = [r for r in rows if r["size_class"] == cls]
        rnd.shuffle(rs)
        for k, r in enumerate(rs):
            base = os.path.basename(r["path"])[:-4]
            if k < 5:
                mix.append(f"{D}/src/{base}.png")
            elif k < 8:
                wj.append((r["path"], f"{D}/src/{base}.webp", k == 7))
                mix.append(wj[-1][1])
            else:
                mix.append(r["path"])
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(to_png, pj))
        list(ex.map(to_webp, wj))
    rep("jpeg", jpgs, 1400)
    rep("png", [d for _, d in pj], 950)
    rep("mix", sorted(mix, key=os.path.basename), 950)
    small = sorted(jpgs, key=mp)[:3]
    open(f"{D}/fewjpg.txt", "w").write("".join(p + "\n" for p in small))
    open(f"{D}/fewpng.txt", "w").write("".join(f"{D}/src/{os.path.basename(p)[:-4]}.png\n" for p in small))


main()
