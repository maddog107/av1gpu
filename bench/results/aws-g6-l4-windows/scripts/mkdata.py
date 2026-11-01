#!/usr/bin/env python3
"""Test data for the Windows vs Linux identity runs (Linux instance), self-contained under ~/data with relative
paths in every list, so the same lists work on both systems (run from ~/data / C:\\data).

jpg/   the 100 size-range JPEGs                       jpg.txt, jpg3.txt (x3: speed)
png40/ the 40 photos of 5-24 MP as RGB PNGs           png40.txt, png40x2.txt (speed)
mix/   25 PNG + 15 WebP (3 lossless) + 60 JPEG        mix.txt, mix3.txt (speed)
alpha/ 39 generated RGBA PNGs + 4 extras              alpha.txt
few/   3 small PNGs; ll.txt: lossless set (10 big PNGs, few, alpha PNGs)
bad/   unreadable / corrupt / unsupported inputs       bad.txt, bad_only.txt, bad_ll.txt
dirinput/  a directory input: images, a text file, a subdirectory named like an image
uni/   non-ASCII file names                             uni.txt
sub12.txt: 12 mixed images (slow preset)
"""
import csv, glob, os, random, shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

H = os.path.expanduser("~")
SRC = f"{H}/images_gamut"
D = f"{H}/data"
MAN = f"{H}/av1gpu/bench/openimages_v4_gamut.csv"


def to_png(a):
    src, dst = a
    im = Image.open(src)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(dst)


def to_webp(a):
    src, dst, lossless = a
    Image.open(src).convert("RGB").save(dst, "WEBP", quality=90, lossless=lossless, method=4)


def wl(name, items):
    open(f"{D}/{name}", "w", newline="\n").write("".join(p + "\n" for p in items))


def main():
    os.chdir(D)
    rows = list(csv.DictReader(open(MAN)))
    for r in rows:
        r["path"] = os.path.join(SRC, r["file"])
        r["mp"] = int(r["width"]) * int(r["height"]) / 1e6
    for sub in ("jpg", "png40", "mix", "alpha", "bad", "few", "dirinput", "dirinput/sub.jpg", "uni"):
        os.makedirs(sub, exist_ok=True)
    for r in rows:
        shutil.copy(r["path"], f"jpg/{r['file']}")
    wl("jpg.txt", sorted(glob.glob("jpg/*.jpg")))
    wl("jpg3.txt", sorted(glob.glob("jpg/*.jpg")) * 3)
    jobs_png, jobs_webp = [], []
    big = [r for r in rows if r["size_class"] in ("large", "huge")]
    for r in big:
        jobs_png.append((r["path"], f"png40/{r['file'][:-4]}.png"))
    rnd = random.Random(15)
    mix = []
    for cls in ("tiny", "small", "typical", "large", "huge"):
        rs = [r for r in rows if r["size_class"] == cls]
        rnd.shuffle(rs)
        for k, r in enumerate(rs):
            base = r["file"][:-4]
            if k < 5:
                dst = f"mix/{base}.png"; jobs_png.append((r["path"], dst))
            elif k < 8:
                dst = f"mix/{base}.webp"; jobs_webp.append((r["path"], dst, k == 7))
            else:
                dst = f"mix/{r['file']}"; shutil.copy(r["path"], dst)
            mix.append(dst)
    with ProcessPoolExecutor(16) as ex:
        list(ex.map(to_png, jobs_png))
        list(ex.map(to_webp, jobs_webp))
    wl("png40.txt", sorted(glob.glob("png40/*.png")))
    wl("png40x2.txt", sorted(glob.glob("png40/*.png")) * 2)
    wl("mix.txt", sorted(mix))
    wl("mix3.txt", sorted(mix) * 3)
    wl("sub12.txt", sorted(mix)[::9][:12])
    for p in sorted(glob.glob(f"{H}/rgba/set/*")) + sorted(glob.glob(f"{H}/rgba/extra/*")):
        shutil.copy(p, "alpha/")
    wl("alpha.txt", sorted(glob.glob("alpha/*")))
    tiny = [r for r in rows if r["size_class"] == "tiny"][:3]
    for r in tiny:
        to_png((r["path"], f"few/{r['file'][:-4]}.png"))
    wl("few.txt", sorted(glob.glob("few/*.png")))
    wl("ll.txt", sorted(glob.glob("png40/*.png"))[:10] + sorted(glob.glob("few/*.png")) + sorted(glob.glob("alpha/*.png")))
    # bad inputs
    B = "bad"
    good_png = sorted(glob.glob("png40/*.png"))[:6]
    some_jpg = sorted(f"jpg/{r['file']}" for r in rows if r["size_class"] == "typical")[:6]
    open(f"{B}/empty.png", "wb").close()
    open(f"{B}/text.png", "w").write("this is not an image\n")
    open(f"{B}/text.jpg", "w").write("this is not a JPEG either\n")
    data = open(good_png[0], "rb").read()
    open(f"{B}/trunc.png", "wb").write(data[: len(data) // 3])
    open(f"{B}/garbage_after_sig.png", "wb").write(data[:8] + random.Random(5).randbytes(5000))
    w = open(sorted(p for p in mix if p.endswith(".webp"))[0], "rb").read()
    open(f"{B}/trunc.webp", "wb").write(w[: len(w) // 2])
    j = open(some_jpg[0], "rb").read()
    open(f"{B}/trunc.jpg", "wb").write(j[: len(j) // 2])
    im = Image.open(some_jpg[1]).convert("RGB")
    im.save(f"{B}/image.gif"); im.save(f"{B}/image.bmp"); im.save(f"{B}/image.tif")
    im.convert("CMYK").save(f"{B}/cmyk.jpg", quality=90)
    shutil.copy(good_png[1], f"{B}/png_named.jpg")
    shutil.copy(some_jpg[2], f"{B}/jpeg_named.png")
    g = (np.add.outer(np.arange(6000) % 256, np.arange(6100) % 256) // 2).astype(np.uint8)
    Image.fromarray(np.stack([g, g[::-1], g[:, ::-1]], -1)).save(f"{B}/huge_36mp.png", compress_level=1)
    a16 = (np.random.RandomState(1).randint(0, 4096, (300, 400)) * 16).astype(np.uint16)
    Image.fromarray(a16).save(f"{B}/gray16_12bit.png")
    bad = [f"{B}/{n}" for n in ("empty.png", "text.png", "text.jpg", "trunc.png", "garbage_after_sig.png", "trunc.webp",
           "trunc.jpg", "image.gif", "image.bmp", "image.tif", "cmyk.jpg", "png_named.jpg", "jpeg_named.png",
           "huge_36mp.png", "gray16_12bit.png", "does_not_exist.png", "does_not_exist.jpg")] + [B]
    lst = []
    for k, p in enumerate(bad):
        lst.append(p)
        if k < len(good_png): lst.append(good_png[k])
        if k < len(some_jpg) - 3: lst.append(some_jpg[k + 3])
    wl("bad.txt", lst)
    wl("bad_only.txt", bad)
    j422 = sorted(f"jpg/{r['file']}" for r in rows if r["subsampling"] == "422")[:4]
    ll = []
    for k, p in enumerate(j422 + [f"{B}/gray16_12bit.png", f"{B}/trunc.png", f"{B}/empty.png"]):
        ll.append(p)
        if k < len(good_png): ll.append(good_png[k])
    wl("bad_ll.txt", ll)
    # directory input: 6 JPEG, 2 PNG, 1 WebP, a text file, an upper-case extension, a dot file, a subdirectory
    for p in sorted(p for p in mix if p.endswith(".jpg"))[:6] + sorted(p for p in mix if p.endswith(".png"))[:2] + \
             sorted(p for p in mix if p.endswith(".webp"))[:1]:
        shutil.copy(p, "dirinput/")
    open("dirinput/notes.txt", "w").write("not an image\n")
    shutil.copy(some_jpg[4], "dirinput/UPPER.JPG")
    shutil.copy(some_jpg[5], "dirinput/.hidden.jpg")
    # non-ASCII names
    shutil.copy(some_jpg[3], "uni/café_ü.jpg")
    shutil.copy(good_png[2], "uni/日本語.png")
    wl("uni.txt", ["uni/café_ü.jpg", "uni/日本語.png"])
    for f in ("jpg", "png40", "mix", "alpha", "few", "ll", "bad", "bad_only", "bad_ll", "uni", "sub12"):
        print(f, len(open(f"{f}.txt", encoding="utf-8").read().split("\n")) - 1)


if __name__ == "__main__":
    main()
