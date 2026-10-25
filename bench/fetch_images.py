#!/usr/bin/env python3
"""Download the benchmark images listed in a manifest and verify their SHA-256.

  bench/fetch_images.py [--manifest bench/openimages_v4_validation_100.csv] [--out bench/images]

The default manifest is 100 JPEGs from the Open Images Dataset V4 validation set (CC BY 2.0; author,
title and source page per image are in the CSV), served by the dataset's public S3 bucket at
https://open-images-dataset.s3.amazonaws.com/validation/<image_id>.jpg.  Files already present
with the right hash are skipped.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(row, out):
    path = os.path.join(out, row["image_id"] + ".jpg")
    if os.path.exists(path) and sha256(path) == row["sha256"]:
        return path, "cached"
    tmp = path + ".part"
    urllib.request.urlretrieve(URL.format(row["image_id"]), tmp)
    got = sha256(tmp)
    if got != row["sha256"]:
        os.remove(tmp)
        raise RuntimeError(f"{row['image_id']}: sha256 {got} != {row['sha256']}")
    os.replace(tmp, path)
    return path, "downloaded"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(HERE, "openimages_v4_validation_100.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "images"))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest, newline="")))
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(fetch, r, a.out): r for r in rows}
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:  # report every failure, then exit non-zero
                print(e, file=sys.stderr)
                bad += 1
    mp = sum(int(r["width"]) * int(r["height"]) for r in rows) / 1e6
    print(f"{len(rows) - bad}/{len(rows)} images ({mp:.1f} MP) in {a.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
