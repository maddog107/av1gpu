#!/usr/bin/env python3
"""Build bench/openimages_v4_gamut.csv: Open Images V4 validation JPEGs spread over the size classes of
docs/RESULTS.md (tiny < 0.3 MP, small < 1 MP, typical < 4 MP, large < 9.4 MP, huge <= 35.6 MP).

  bench/make_gamut_manifest.py [--per-class 20] [--seed 1] [--candidates 600]

The 100-image manifest uses the dataset's 1024-pixel copies, so every image is about 1 MP.  This one
takes the original uploads (OriginalURL on Flickr, checked against the dataset's OriginalMD5) for the
three large classes, Thumbnail300KURL for tiny, and the dataset's 1024-pixel S3 copy for small.  Rows are
drawn in a seeded random order; the first --per-class usable JPEGs of each class are kept (8-bit, RGB or
grayscale, at most 35.6 MP).  Only needs to be rerun to change the set; bench/fetch_images.py
--manifest bench/openimages_v4_gamut.csv downloads it.  All images are CC BY 2.0.
"""
import argparse
import base64
import concurrent.futures as cf
import csv
import hashlib
import io
import os
import random
import urllib.request

from PIL import Image, JpegImagePlugin

HERE = os.path.dirname(os.path.abspath(__file__))
META = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
S3 = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
CLASSES = [("tiny", 0, 0.3), ("small", 0.3, 1.0), ("typical", 1.0, 4.0), ("large", 4.0, 9.4), ("huge", 9.4, 35.6)]
ALLOWED = dict(thumb=("tiny",), s3=("small",), original=("typical", "large", "huge"))
SUB = {0: "444", 1: "422", 2: "420"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "av1gpu-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def probe(data):
    im = Image.open(io.BytesIO(data))
    if im.format != "JPEG" or im.mode not in ("RGB", "L"):
        return None
    sub = "gray" if im.mode == "L" else SUB.get(JpegImagePlugin.get_sampling(im), "other")
    return dict(width=im.size[0], height=im.size[1], subsampling=sub, icc=int(bool(im.info.get("icc_profile"))),
                progressive=int(bool(im.info.get("progressive"))), orientation=im.getexif().get(0x0112, 1))


def candidate(row, kind):
    try:
        if kind == "original":
            data = get(row["OriginalURL"])
            if base64.b64encode(hashlib.md5(data).digest()).decode() != row["OriginalMD5"]:
                return None    # removed from Flickr or replaced by a placeholder
            url = row["OriginalURL"]
        else:
            url = row["Thumbnail300KURL"] if kind == "thumb" else S3.format(row["ImageID"])
            data = get(url)
        p = probe(data)
    except Exception:
        return None
    if p is None:
        return None
    mp = p["width"] * p["height"] / 1e6
    cls = next((c for c, lo, hi in CLASSES if lo <= mp < hi or (c == "huge" and mp == hi)), None)
    if cls not in ALLOWED[kind]:
        return None
    return dict(image_id=row["ImageID"], file=f"{row['ImageID']}_{cls}.jpg", size_class=cls, url=url,
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), **p, author=row["Author"],
                title=row["Title"], landing_url=row["OriginalLandingURL"], license=row["License"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=600, help="originals to try for the three large classes")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "openimages_v4_gamut.csv"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(io.StringIO(get(META).decode())) if "/by/2.0/" in r["License"]]
    used = {r["image_id"] for r in csv.DictReader(open(os.path.join(HERE, "openimages_v4_validation_100.csv")))}
    rows = [r for r in rows if r["ImageID"] not in used]
    random.Random(a.seed).shuffle(rows)

    # disjoint row ranges per source, so no photo appears twice
    n = a.per_class * 3
    jobs = [(r, "thumb") for r in rows[:n]] + [(r, "s3") for r in rows[n:2 * n]] + \
           [(r, "original") for r in rows[2 * n:2 * n + a.candidates]]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        found = list(ex.map(lambda j: candidate(*j), jobs))   # keeps the seeded order
    keep = {c: [] for c, _, _ in CLASSES}
    for f in found:
        if f and len(keep[f["size_class"]]) < a.per_class:
            keep[f["size_class"]].append(f)
    out = [f for c, _, _ in CLASSES for f in keep[c]]
    cols = ["file", "image_id", "size_class", "url", "sha256", "width", "height", "bytes", "subsampling", "icc",
            "progressive", "orientation", "author", "title", "landing_url", "license"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    for c, _, _ in CLASSES:
        mp = sum(f["width"] * f["height"] for f in keep[c]) / 1e6
        print(f"{c:8s} {len(keep[c]):3d} images {mp:7.1f} MP")
    print(f"{len(out)} images -> {a.out}")


if __name__ == "__main__":
    main()
