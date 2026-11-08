#!/usr/bin/env python3
"""Download the benchmark images listed in a manifest and verify their SHA-256.

  bench/fetch_images.py [--manifest bench/openimages_v4_validation_100.csv] [--out bench/images]

The default manifest is 100 JPEGs from the Open Images Dataset V4 validation set (CC BY 2.0; author,
title and source page per image are in the CSV), served by the dataset's public S3 bucket at
https://open-images-dataset.s3.amazonaws.com/validation/<image_id>.jpg.  A manifest may also give
`file` (local name) and `url` columns, as bench/openimages_v4_gamut.csv does.  Files already present
with the right hash are skipped.

Flickr writes a Photoshop / IPTC APP13 segment with a per-download link (https://flickr.com/e/...) into the
files it serves, so a photo's bytes can change from one download to the next while its picture and the metadata
av1gpu reads (EXIF, ICC) stay the same.  A manifest's `content_sha256` column is the hash without APP13
segments: a file that matches it is accepted too.
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


def content_sha256(data):
    """SHA-256 of a JPEG without its APP13 segments (the whole file if it isn't a JPEG)."""
    if data[:2] != b"\xff\xd8":
        return hashlib.sha256(data).hexdigest()
    h, i = hashlib.sha256(data[:2]), 2
    while i + 4 <= len(data) and data[i] == 0xFF and 0xC0 <= data[i + 1] <= 0xFE and data[i + 1] not in (0xD8, 0xD9, 0xDA):
        end = i + 2 + int.from_bytes(data[i + 2:i + 4], "big")
        if data[i + 1] != 0xED:
            h.update(data[i:end])
        i = end
    h.update(data[i:])   # from the first scan (or anything unexpected) on, as is
    return h.hexdigest()


def matches(data, row):
    return hashlib.sha256(data).hexdigest() == row["sha256"] or \
        bool(row.get("content_sha256")) and content_sha256(data) == row["content_sha256"]


def fetch(row, out):
    path = os.path.join(out, row.get("file") or row["image_id"] + ".jpg")
    if os.path.exists(path) and matches(open(path, "rb").read(), row):
        return path, "cached"
    tmp = path + ".part"
    urllib.request.urlretrieve(row.get("url") or URL.format(row["image_id"]), tmp)
    data = open(tmp, "rb").read()
    if not matches(data, row):
        os.remove(tmp)
        raise RuntimeError(f"{row['image_id']}: sha256 {hashlib.sha256(data).hexdigest()} != {row['sha256']}"
                           + (f", without APP13 {content_sha256(data)} != {row['content_sha256']}"
                              if row.get("content_sha256") else ""))
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
