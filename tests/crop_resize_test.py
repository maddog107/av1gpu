#!/usr/bin/env python3
"""--crop / --max-size / --max-height and the per-line options of --batch lists.

On a PNG (and an RGBA PNG made here): output sizes follow ImageMagick's -crop and -resize 'WxH>' geometry; a crop of
the whole image and a box larger than it give the bytes of a plain encode; with --lossless a crop decodes to exactly
the crop of the lossless encode; shrinking an image with transparency keeps the colour of its visible pixels (alpha-
weighted taps: no bleeding of the colour under transparent pixels into the edges).  Errors: a crop outside the image
(status 3), an empty crop and bad syntax (status 2).  A --batch list with crop= / max-size= lines, a file name with a
TAB, a line with a bad option and one with a crop outside its image: the good lines are encoded at their sizes, the
other two are skipped, and the run exits 0.

  tests/crop_resize_test.py --bin build/av1gpu --image tests/gate_images/cfl_edge_211x149.png
"""
import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib


def ispe(path):
    """(width, height) of the first ispe box of an AVIF"""
    d = open(path, "rb").read()
    i = d.find(b"ispe")
    return struct.unpack(">II", d[i + 8:i + 16]) if i >= 0 else None


def png_size(path):
    return struct.unpack(">II", open(path, "rb").read()[16:24])


def im_geometry(w, h, crop=None, box=None):
    """ImageMagick: -crop WxH+X+Y (clipped to the image), then -resize 'WxH>' (box: (W or 0, H or 0))"""
    if crop:
        cw, ch, cx, cy = crop
        w, h = min(cx + cw, w) - cx, min(cy + ch, h) - cy
    if box:
        s = min([1.0] + [b / v for b, v in zip(box, (w, h)) if b])
        if s < 1:
            w, h = max(1, math.floor(w * s + 0.5)), max(1, math.floor(h * s + 0.5))
    return w, h


def write_rgba(path, w, h):
    """transparent black, with an opaque white rectangle in the middle"""
    rows = []
    for y in range(h):
        r = bytearray()
        for x in range(w):
            inside = w // 4 <= x < 3 * w // 4 and h // 4 <= y < 3 * h // 4
            r += bytes((255, 255, 255, 255)) if inside else bytes((0, 0, 0, 0))
        rows.append(b"\x00" + bytes(r))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_crop_")
    fails = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    def run(*args):
        return subprocess.run([a.bin] + [str(x) for x in args], capture_output=True, text=True, timeout=300)

    try:
        W, H = png_size(a.image)
        img = os.path.join(t, "src.png")
        shutil.copy(a.image, img)
        plain = os.path.join(t, "plain.avif")
        p = run(img, plain)
        check("plain encode", p.returncode == 0, p.stderr.strip()[-200:])
        # ---- sizes (single-image mode) ----
        for opts, crop, box in ((["--crop", "100x50+11+7"], (100, 50, 11, 7), None),
                                (["--crop", "11,7,100,50"], (100, 50, 11, 7), None),
                                (["--crop", "300x300+150+100"], (300, 300, 150, 100), None),   # clipped
                                (["--max-size", "100x100"], None, (100, 100)),
                                (["--max-size", "60x"], None, (60, 0)),
                                (["--max-height", "33"], None, (0, 33)),
                                (["--crop", "101x51+1+1", "--max-height", "20"], (101, 51, 1, 1), (0, 20)),
                                (["--crop", "1x1+5+5"], (1, 1, 5, 5), None),
                                (["--crop", "3x5+0+0", "--max-size", "2x2"], (3, 5, 0, 0), (2, 2))):
            o = os.path.join(t, "s.avif")
            p = run(img, o, *opts)
            want = im_geometry(W, H, crop, box)
            got = ispe(o) if p.returncode == 0 else None
            check(f"size {' '.join(opts)}", got == want, f"{got} (want {want}) {p.stderr.strip()[-200:]}")
        for opts in (["--crop", f"{W}x{H}+0+0"], ["--max-size", f"{W}x{H}"], ["--max-size", "5000x5000"]):
            o = os.path.join(t, "id.avif")
            p = run(img, o, *opts)
            same = p.returncode == 0 and open(o, "rb").read() == open(plain, "rb").read()
            check(f"identity {' '.join(opts)} == plain encode", same)
        # ---- --lossless: a crop is exact ----
        full, part = os.path.join(t, "ll.avif"), os.path.join(t, "llc.avif")
        p1 = run(img, full, "--lossless", "--debug-outputs")
        cx, cy, cw, ch = 13, 9, 50, 40
        p2 = run(img, part, "--lossless", "--debug-outputs", "--crop", f"{cw}x{ch}+{cx}+{cy}")
        ok = p1.returncode == 0 and p2.returncode == 0
        if ok:
            fy, py = open(full + ".recon.yuv", "rb").read(), open(part + ".recon.yuv", "rb").read()
            n = W * H
            want = b"".join(fy[pl * n + (cy + y) * W + cx: pl * n + (cy + y) * W + cx + cw] for pl in range(3) for y in range(ch))
            ok = py == want
        check("--lossless --crop decodes to the crop of the lossless encode", ok)
        # ---- transparency: alpha-weighted colour ----
        rgba = os.path.join(t, "rgba.png")
        write_rgba(rgba, 64, 48)
        dump = os.path.join(t, "rgba.dump")
        p = run(rgba, os.path.join(t, "rgba.avif"), "--max-size", "21x21", "--yuv", "444", "--dump-source", dump)
        ok = p.returncode == 0
        bad = -1
        if ok:
            ow, oh = im_geometry(64, 48, None, (21, 21))
            d = open(dump, "rb").read()
            n = ow * oh
            ok = len(d) == 4 * n
            if ok:   # Y, U, V, alpha (4:4:4 8-bit): wherever alpha is visible the colour is the rectangle's white
                bad = sum(1 for i in range(n) if d[3 * n + i] and d[i] < 254)
                ok = bad == 0 and any(0 < d[3 * n + i] < 255 for i in range(n))
        check("RGBA shrink keeps the colour of visible pixels (alpha-weighted)", ok, f"{bad} pixels darker")
        # ---- errors ----
        p = run(img, os.path.join(t, "e.avif"), "--crop", f"10x10+{W}+0")
        check("crop outside the image: status 3", p.returncode == 3 and "outside" in p.stderr, p.stderr.strip()[-200:])
        for opts, word in ((["--crop", "0x10+0+0"], "empty"), (["--crop", "10x10"], "expected"), (["--crop", "-5,0,10,10"], "expected"),
                           (["--max-size", "0x0"], "at least"), (["--max-size", "100"], "expected"), (["--max-height", "abc"], "height"),
                           (["--max-mp", "zero"], "megapixels")):
            p = run(img, os.path.join(t, "e.avif"), *opts)
            check(f"bad {' '.join(opts)}: status 2", p.returncode == 2 and word in p.stderr,
                  (p.stderr.strip().splitlines() or [""])[0][-200:])
        # ---- batch list options ----
        src = os.path.join(t, "in")
        os.makedirs(src)
        lines = [("a.png\tcrop=100x50+0+0", (100, 50, 0, 0), None), ("b.png\tmax-size=x30\tcrop=5,5,150,100", (150, 100, 5, 5), (0, 30)),
                 ("c.png\tfoo=1", "skip", None), ("d.png\tcrop=10x10+500+500", "skip", None), ("e.png\t\tmax-size=50x50\r", None, (50, 50))]
        if os.name != "nt":   # (no TAB in Windows file names)
            lines.append(("tab\tname.png", None, None))
        for l, _, _ in lines:
            shutil.copy(img, os.path.join(src, l if l == "tab\tname.png" else l.split("\t")[0]))
        lst = os.path.join(t, "list.txt")
        with open(lst, "w", newline="") as f:
            f.write("".join(os.path.join(src, l) + "\n" for l, _, _ in lines))
        out = os.path.join(t, "out")
        p = run("--batch", lst, "--outdir", out)
        m = re.search(r"skipped \(unsupported\): (\d+)", p.stderr)
        check("batch list: exit 0, 2 skipped", p.returncode == 0 and m and int(m.group(1)) == 2, p.stderr.strip()[-400:])
        for l, crop, box in lines:
            n = l.split("\t")[0] if l != "tab\tname.png" else l
            o = os.path.join(out, n + ".avif")
            if crop == "skip":
                check(f"batch list: {l!r} skipped", not os.path.exists(o))
                continue
            want = im_geometry(W, H, crop, box)
            got = ispe(o) if os.path.exists(o) else None
            check(f"batch list: {l!r} size", got == want, f"{got} (want {want})")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
