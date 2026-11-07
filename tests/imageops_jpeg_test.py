#!/usr/bin/env python3
"""The image operations on JPEGs, which reach the GPU through the ingest and not through the PNG decoder.

On a 4:2:0 page of coloured squares in a white margin and a blank page, written by jpeg_gray_gen --pages: geometries are
for the displayed image, also with an EXIF rotation by 90 degrees and on a --batch list line; --trim finds the picture,
a blank page is skipped, and a page after it in a batch keeps its rotation; --negate is exact at 4:2:0, and --grayscale
Rec601Luma keeps the Y plane.  Pictures that can't be encoded are refused cleanly, in single-image mode and in a batch,
without costing the other images: wider than AV1 can code, a geometry that rounds to 0, over --max-mp (the message gives
the --max-mp to use, and using it works).  In a batch --max-mp counts the trimmed size, as in single-image mode.

  tests/imageops_jpeg_test.py --bin build/av1gpu --gen build/jpeg_gray_gen
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile


def ispe(path):
    """(width, height) of the first ispe box of an AVIF; None without the file"""
    if not os.path.exists(path):
        return None
    d = open(path, "rb").read()
    i = d.find(b"ispe")
    return struct.unpack(">II", d[i + 8:i + 16]) if i >= 0 else None


def irot(path):
    """the irot property's angle (in steps of 90 degrees), None without one (or without the file)"""
    if not os.path.exists(path):
        return None
    d = open(path, "rb").read()
    i = d.find(b"irot")
    return d[i + 4] & 3 if i >= 0 else None


def with_orientation(src, dst, orientation):
    """src with an EXIF APP1 segment that says orientation, right after the SOI"""
    tiff = b"MM\x00\x2a\x00\x00\x00\x08" + struct.pack(">HHHIHHI", 1, 0x0112, 3, 1, orientation, 0, 0)
    payload = b"Exif\x00\x00" + tiff
    d = open(src, "rb").read()
    open(dst, "wb").write(d[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + d[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_jpg_")
    fails = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    def run(*args):
        return subprocess.run([a.bin] + [str(x) for x in args], capture_output=True, text=True, timeout=300)

    def size_of(img, *opts):
        o = os.path.join(t, "s.avif")
        if os.path.exists(o):
            os.remove(o)
        p = run(img, o, *opts)
        return (ispe(o) if p.returncode == 0 and os.path.exists(o) else None), p

    def planes(img, *opts):
        d = os.path.join(t, "d.yuv")
        p = run(img, os.path.join(t, "o.avif"), "--dump-source", d, *opts)
        return open(d, "rb").read() if p.returncode == 0 else None

    try:
        subprocess.run([a.gen, "--pages", t], check=True, capture_output=True)
        page, blank = os.path.join(t, "page.jpg"), os.path.join(t, "blank.jpg")
        turned = os.path.join(t, "turned.jpg")
        with_orientation(page, turned, 6)
        W, H = 640, 480
        # ---- geometry on the displayed image: stored 640x480 with orientation 6 is displayed 480x640 ----
        for g, plain, rot in (("50%", (320, 240), (320, 240)), ("150%", (960, 720), (960, 720)), ("x50%", (640, 240), (320, 480)),
                              ("50%x", (320, 240), (320, 240)), ("50%x25%", (320, 120), (160, 240)), ("100x200!", (100, 200), (200, 100)),
                              ("100!", (100, 480), (640, 100)), ("240x", (240, 180), (320, 240)), ("x240", (320, 240), (240, 180))):
            got, _ = size_of(page, "--resize", g)
            check(f"--resize {g}", got == plain, f"{got} (want {plain})")
            got, _ = size_of(turned, "--resize", g)
            check(f"--resize {g}, EXIF 6", got == rot, f"{got} (want {rot})")
        # ---- a batch, with options on the list lines ----
        lst, out = os.path.join(t, "list"), os.path.join(t, "out")
        with open(lst, "w") as f:
            f.write(f"{turned}\tresize=50%\n{page}\tresize=0.4x\n{page}\tresize=70000x64!\n{blank}\tresize=25%\n")
        p = run("--batch", lst, "--outdir", out)
        check("batch: bad lines do not stop the good ones", p.returncode == 0, p.stderr.strip()[-300:])
        check("batch: EXIF 6, resize=50%", ispe(os.path.join(out, "turned.jpg.avif")) == (320, 240), f"{ispe(os.path.join(out, 'turned.jpg.avif'))}")
        check("batch: resize=25%", ispe(os.path.join(out, "blank.jpg.avif")) == (160, 120))
        check("batch: resize=0.4x named", "rounds to 0" in p.stderr, p.stderr.strip()[-300:])
        check("batch: a width above AV1's limit named", "larger than AV1 can code" in p.stderr, p.stderr.strip()[-300:])
        check("batch: those wrote nothing", not os.path.exists(os.path.join(out, "page.jpg.avif")))
        # ---- trim ----
        got, _ = size_of(page, "--trim", "--fuzz", "5%")
        check("--trim: the picture", got == (320, 240), f"{got}")
        got, _ = size_of(turned, "--trim", "--fuzz", "5%", "--resize", "50%")
        check("--trim --resize 50%, EXIF 6", got == (160, 120), f"{got}")
        got, p = size_of(blank, "--trim", "--skip-blank", "99")
        check("blank JPEG: status 3", got is None and p.returncode == 3 and "blank" in p.stderr, p.stderr.strip()[-200:])
        lst2, out2 = os.path.join(t, "list2"), os.path.join(t, "out2")
        with open(lst2, "w") as f:
            f.write(f"{blank}\n{turned}\n")
        p = run("--batch", lst2, "--outdir", out2, "--trim", "--fuzz", "5%", "--skip-blank", "99")
        o = os.path.join(out2, "turned.jpg.avif")
        check("batch: blank JPEG skipped, the next page keeps its size and rotation",
              p.returncode == 0 and not os.path.exists(os.path.join(out2, "blank.jpg.avif")) and os.path.exists(o) and ispe(o) == (320, 240) and irot(o) is not None,
              p.stderr.strip()[-300:])
        # ---- 4:2:0 samples ----
        base, neg = planes(page), planes(page, "--negate")
        cn = 2 * (W // 2) * (H // 2)
        ok = base is not None and neg is not None and len(base) == len(neg) == W * H + cn
        badY = sum(1 for i in range(W * H) if neg[i] != 255 - base[i]) if ok else -1
        badC = max(abs(neg[W * H + i] - min(255, 256 - base[W * H + i])) for i in range(cn)) if ok else -1
        check("--negate at 4:2:0: Y exact, chroma within 1", ok and badY == 0 and badC <= 1, f"{badY} Y samples differ, chroma off by up to {badC}")
        # a crop at even offsets keeps the JPEG's own chroma: gray of the crop = the crop of the gray (but at the crop's edges,
        # where the upsampling no longer sees the samples outside it)
        full, part = planes(page, "--grayscale", "Rec709Luma"), planes(page, "--grayscale", "Rec709Luma", "--crop", "320x240+64+48")
        ok = full is not None and part is not None and len(part) == 320 * 240
        bad = sum(1 for y in range(1, 239) for x in range(1, 319) if part[y * 320 + x] != full[(48 + y) * W + 64 + x]) if ok else -1
        check("--grayscale after an even crop: the crop of the gray", ok and bad == 0, f"{bad} samples differ")
        gray = planes(page, "--grayscale", "Rec601Luma")
        check("--grayscale Rec601Luma keeps the Y plane", gray is not None and base is not None and gray == base[:W * H])
        # ---- the size limits ----
        got, p = size_of(page, "--resize", "70000x64!")
        check("--resize 70000x64!: status 3, nothing written", got is None and p.returncode == 3 and "larger than AV1 can code" in p.stderr, p.stderr.strip()[-200:])
        got, p = size_of(page, "--resize", "1x65473!")
        check("--resize 1x65473!: status 3, the rows named", got is None and p.returncode == 3 and "taller" in p.stderr, p.stderr.strip()[-200:])
        got, p = size_of(page, "--max-mp", "0.1")
        m = re.search(r"larger than 0\.1 megapixels.*add --max-mp (\d+)", p.stderr)
        check("over --max-mp: status 3, the message says what to add", got is None and p.returncode == 3 and m is not None, p.stderr.strip()[-200:])
        if m:
            got, _ = size_of(page, "--max-mp", m.group(1))
            check("the --max-mp it suggests works", got == (W, H), f"{got}")
        got, p = size_of(page, "--max-mp", "0.1", "--max-input-mp", "1", "--resize", "200%")
        m = re.search(r"1280x960 after crop / resize is larger than 0\.1 megapixels \(--max-mp\); add --max-mp 2 ", p.stderr)
        check("over --max-mp after the resize, before any GPU work on it", got is None and p.returncode == 3 and m is not None, p.stderr.strip()[-200:])
        out3 = os.path.join(t, "out3")
        lst3 = os.path.join(t, "list3")
        with open(lst3, "w") as f:
            f.write(f"{page}\n")
        p = run("--batch", lst3, "--outdir", out3, "--max-mp", "0.1")
        check("batch, over --max-mp: skipped, status 0, the message says what to add",
              p.returncode == 0 and re.search(r"skipped \S*page\.jpg: larger than 0\.1 megapixels.*add --max-mp 1", p.stderr) is not None, p.stderr.strip()[-300:])
        # --max-mp counts what is encoded: 640x480 is 0.31 MP, its picture 0.08
        got, p = size_of(page, "--trim", "--fuzz", "5%", "--max-mp", "0.1", "--max-input-mp", "1")
        check("single image: --max-mp after the trim", got == (320, 240), f"{got} {p.stderr.strip()[-200:]}")
        p = run("--batch", lst3, "--outdir", out3, "--trim", "--fuzz", "5%", "--max-mp", "0.1", "--max-input-mp", "1")
        o = os.path.join(out3, "page.jpg.avif")
        check("batch: --max-mp after the trim", p.returncode == 0 and os.path.exists(o) and ispe(o) == (320, 240), p.stderr.strip()[-300:])
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
