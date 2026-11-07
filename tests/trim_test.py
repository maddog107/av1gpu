#!/usr/bin/env python3
"""--trim, --fuzz and the trim conditions (--trim-margin, --trim-limits, --trim-lightness, --skip-blank).

On PNGs made here: a page with a white margin around a grey picture is trimmed to the picture, in 4:2:0 and 4:4:4,
at 8 and 10 bits, and with --lossless, where the trimmed encode decodes to exactly the crop of the lossless encode;
--fuzz lets a paper margin with specks in it go and keeps a picture that differs by more; an image without a border
of one colour, and a blank one without --skip-blank, give the bytes of a plain encode; the margin, limits, lightness
and blank conditions decide as described in --help; a page of two colours with nothing on it is not blank; --crop is
applied first, also at an offset, --max-size after; a transparent border around an opaque picture is trimmed.  Errors: the trim options without --trim and bad values (status 2), a blank image
with --skip-blank (status 3).  A --batch run with a blank image skips it with a message and counts the trims.  With
ImageMagick (`magick`) installed, the trimmed sizes are also compared with `magick -fuzz F -format %@ info:`.

  tests/trim_test.py --bin build/av1gpu
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


def write_png(path, w, h, px, alpha=False):
    """px(x, y) -> (r, g, b) or (r, g, b, a), 8 bits"""
    n = 4 if alpha else 3
    rows = []
    for y in range(h):
        r = bytearray(b"\x00")
        for x in range(w):
            r += bytes(px(x, y)[:n])
        rows.append(bytes(r))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def page(w, h, box, paper=(255, 255, 255), ink=(60, 60, 60), specks=()):
    """a w x h page of paper with a picture (a grey checker pattern) in box = (x, y, bw, bh), and specks of colour"""
    bx, by, bw, bh = box
    sp = dict(specks)

    def px(x, y):
        if (x, y) in sp:
            return sp[(x, y)]
        if bx <= x < bx + bw and by <= y < by + bh:
            return ink if ((x - bx) // 4 + (y - by) // 4) % 2 else (ink[0] + 40, ink[1] + 40, ink[2] + 40)
        return paper
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_trim_")
    fails = []
    magick = shutil.which("magick")

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
        return (ispe(o) if p.returncode == 0 else None), p

    def im_box(img, fuzz):
        out = subprocess.run([magick, img, "-fuzz", fuzz, "-format", "%@", "info:"], capture_output=True, text=True).stdout
        m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", out)
        return tuple(int(v) for v in m.groups()) if m else None

    try:
        W, H = 240, 160
        box = (31, 17, 150, 101)   # odd offsets and sizes
        img = os.path.join(t, "page.png")
        write_png(img, W, H, page(W, H, box))
        plain = os.path.join(t, "plain.avif")
        check("plain encode", run(img, plain).returncode == 0)
        # ---- the box ----
        for opts in ([], ["--yuv", "444"], ["--depth", "10"], ["--yuv", "444", "--depth", "10"]):
            got, p = size_of(img, "--trim", *opts)
            check(f"--trim {' '.join(opts)}: the picture's size", got == box[2:], f"{got} (want {box[2:]}) {p.stderr.strip()[-200:]}")
        full, part = os.path.join(t, "ll.avif"), os.path.join(t, "llt.avif")
        p1 = run(img, full, "--lossless", "--debug-outputs")
        p2 = run(img, part, "--lossless", "--debug-outputs", "--trim")
        ok = p1.returncode == 0 and p2.returncode == 0
        if ok:
            fy, py = open(full + ".recon.yuv", "rb").read(), open(part + ".recon.yuv", "rb").read()
            bx, by, bw, bh = box
            n = W * H
            ok = py == b"".join(fy[pl * n + (by + y) * W + bx: pl * n + (by + y) * W + bx + bw] for pl in range(3) for y in range(bh))
        check("--lossless --trim decodes to the box of the lossless encode", ok)
        # ---- --fuzz: specks in the margin ----
        specky = os.path.join(t, "specky.png")
        write_png(specky, W, H, page(W, H, box, specks=[((5, 5), (235, 235, 235)), ((W - 3, H - 9), (225, 230, 240))]))
        got, _ = size_of(specky, "--trim", "--yuv", "444")
        check("--trim without --fuzz keeps the specks", got not in (None, box[2:]), f"{got}")
        got, _ = size_of(specky, "--trim", "--fuzz", "10%", "--yuv", "444")
        check("--fuzz 10% lets the specks go", got == box[2:], f"{got} (want {box[2:]})")
        got, _ = size_of(specky, "--trim", "--fuzz", "6553", "--yuv", "444")   # Q16 units: 6553 / 65535 = 10 %
        check("--fuzz in Q16 units", got == box[2:], f"{got} (want {box[2:]})")
        # (ImageMagick's threshold: grey 37 steps from white differs at 10 %, 36 does not)
        for v, differs in ((255 - 36, False), (255 - 37, True)):
            g = os.path.join(t, f"grey{v}.png")
            write_png(g, 40, 40, lambda x, y, v=v: (v, v, v) if 10 <= x < 30 and 10 <= y < 30 else (255, 255, 255))
            got, _ = size_of(g, "--trim", "--fuzz", "10%", "--yuv", "444")
            check(f"--fuzz 10%: grey {v} {'differs' if differs else 'is border'}", got == ((20, 20) if differs else (40, 40)), f"{got}")
        if magick:
            for f, im in (("10%", specky), ("0%", specky), ("10%", img)):
                want = im_box(im, f)
                got, _ = size_of(im, "--trim", "--fuzz", f, "--yuv", "444")
                check(f"ImageMagick -fuzz {f} -trim size", want is not None and got == want[:2], f"{got} (ImageMagick {want})")
        # ---- identities ----
        noborder = os.path.join(t, "noborder.png")
        write_png(noborder, W, H, page(W, H, (0, 0, W, H)))
        p = run(noborder, os.path.join(t, "nb.avif"), "--trim")
        nbplain = os.path.join(t, "nbplain.avif")
        run(noborder, nbplain)
        check("no border: --trim == plain encode", p.returncode == 0 and open(os.path.join(t, "nb.avif"), "rb").read() == open(nbplain, "rb").read())
        blank = os.path.join(t, "blank.png")
        write_png(blank, 64, 48, lambda x, y: (250, 250, 250))
        bplain = os.path.join(t, "bplain.avif")
        run(blank, bplain)
        p = run(blank, os.path.join(t, "b.avif"), "--trim")
        check("blank without --skip-blank: encoded whole", p.returncode == 0 and open(os.path.join(t, "b.avif"), "rb").read() == open(bplain, "rb").read())
        p = run(blank, os.path.join(t, "b2.avif"), "--trim", "--skip-blank", "99")
        check("blank with --skip-blank: status 3, nothing written", p.returncode == 3 and "blank" in p.stderr and not os.path.exists(os.path.join(t, "b2.avif")),
              p.stderr.strip()[-200:])
        two = os.path.join(t, "two.png")   # edges that cross: no box, and nothing that says the page is blank
        write_png(two, 64, 48, lambda x, y: (0, 0, 0) if x < 32 else (255, 255, 255))
        got, p = size_of(two, "--trim", "--skip-blank", "99")
        check("two-colour page with --skip-blank: encoded whole", got == (64, 48), f"{got} {p.stderr.strip()[-200:]}")
        # ---- conditions ----
        bx, by, bw, bh = box
        m = min(W, H) * 2 // 100   # --trim-margin 2%
        got, _ = size_of(img, "--trim", "--trim-margin", "2%")
        want = (min(W, bx + bw + m) - max(0, bx - m), min(H, by + bh + m) - max(0, by - m))
        check("--trim-margin 2%", got == want, f"{got} (want {want})")
        got, _ = size_of(img, "--trim", "--trim-margin", "40")
        want = (min(W, bx + bw + 40) - max(0, bx - 40), min(H, by + bh + 40) - max(0, by - 40))
        check("--trim-margin 40 (clipped to the image)", got == want, f"{got} (want {want})")
        # the box removes 90 of 240 columns (37 %) and 59 of 160 rows (36 %)
        got, _ = size_of(img, "--trim", "--trim-limits", "1%,20%")
        check("--trim-limits 1%,20%: too much, whole", got == (W, H), f"{got}")
        got, _ = size_of(img, "--trim", "--trim-limits", "1%,40%")
        check("--trim-limits 1%,40%: trimmed", got == box[2:], f"{got}")
        got, _ = size_of(img, "--trim", "--trim-limits", "37%,40%")   # 59 rows >= floor(160 * 37 %) = 59
        check("--trim-limits 37%,40%: trimmed", got == box[2:], f"{got}")
        got, _ = size_of(img, "--trim", "--trim-limits", "38%,40%")   # 59 rows < floor(160 * 38 %) = 60
        check("--trim-limits 38%,40%: too little off the height, whole", got == (W, H), f"{got}")
        dark = os.path.join(t, "dark.png")
        write_png(dark, W, H, page(W, H, box, paper=(40, 30, 30), ink=(160, 160, 160)))
        got, _ = size_of(dark, "--trim", "--trim-lightness", "80")
        check("--trim-lightness 80: dark border, whole", got == (W, H), f"{got}")
        got, _ = size_of(img, "--trim", "--trim-lightness", "80")
        check("--trim-lightness 80: white border, trimmed", got == box[2:], f"{got}")
        onecorner = os.path.join(t, "onecorner.png")   # one dark corner is outvoted; it is the picture running off the edge
        write_png(onecorner, W, H, page(W, H, box, specks=[((0, 0), (0, 0, 0))]))
        got, _ = size_of(onecorner, "--trim", "--trim-lightness", "80")
        check("--trim-lightness: one dark corner is outvoted", got is not None and got != (W, H), f"{got}")
        # ---- order: --crop first, --max-size after ----
        got, _ = size_of(img, "--trim", "--crop", "100x100+0+0")
        check("--crop then --trim", got == (100 - bx, 100 - by), f"{got} (want {(100 - bx, 100 - by)})")
        got, _ = size_of(img, "--trim", "--crop", "120x110+20+10")   # the box is 31..180 x 17..117, the crop 20..139 x 10..119
        check("--crop at an offset, then --trim", got == (140 - 31, 118 - 17), f"{got} (want {(140 - 31, 118 - 17)})")
        got, _ = size_of(img, "--trim", "--max-size", "75x")
        check("--trim then --max-size", got == (75, math.floor(bh * 75 / bw + 0.5)), f"{got}")
        # ---- which corner each edge is measured against ----
        brc = os.path.join(t, "br.png")   # the bottom row is the bottom-left's colour, but for the bottom-right corner
        write_png(brc, 40, 40, lambda x, y: (0, 0, 0) if (x, y) == (39, 39) else (255, 255, 255) if y < 39 or x < 39 else (0, 0, 0))
        got, _ = size_of(brc, "--trim", "--yuv", "444")
        if magick:
            want = im_box(brc, "0%")
            check("bottom-right corner rule: ImageMagick's size", want is not None and got == want[:2], f"{got} (ImageMagick {want})")
        else:
            check("bottom-right corner rule", got == (39, 40), f"{got}")
        # ---- transparency ----
        rgba = os.path.join(t, "rgba.png")
        write_png(rgba, 64, 48, lambda x, y: (200, 10, 10, 255) if 9 <= x < 41 and 5 <= y < 30 else (0, 0, 0, 0), alpha=True)
        got, _ = size_of(rgba, "--trim")
        check("transparent border", got == (32, 25), f"{got}")
        # ---- errors ----
        for opts, word in ((["--fuzz", "10%"], "need --trim"), (["--skip-blank", "99"], "need --trim"),
                           (["--trim", "--fuzz", "abc"], "expected"), (["--trim", "--fuzz", "101%"], "above 100"),
                           (["--trim", "--trim-limits", "20%,1%"], "MIN"), (["--trim", "--trim-limits", "5%"], "MIN"),
                           (["--trim", "--trim-margin", "1.5"], "whole"), (["--trim", "--trim-lightness", "120"], "0..100")):
            p = run(img, os.path.join(t, "e.avif"), *opts)
            check(f"bad {' '.join(opts)}: status 2", p.returncode == 2 and word in p.stderr, (p.stderr.strip().splitlines() or [""])[0][-200:])
        # ---- batch ----
        src = os.path.join(t, "in")
        os.makedirs(src)
        for n in ("page", "blank", "noborder", "dark"):
            shutil.copy(os.path.join(t, n + ".png"), src)
        out = os.path.join(t, "out")
        p = run("--batch", src, "--outdir", out, "--trim", "--trim-lightness", "80", "--skip-blank", "99")
        check("batch: exit 0", p.returncode == 0, p.stderr.strip()[-400:])
        check("batch: the blank image skipped with a message",
              not os.path.exists(os.path.join(out, "blank.png.avif")) and re.search(r"skipped \S*blank\.png: blank", p.stderr) is not None)
        check("batch: summary", "trim: 1 images trimmed, 1 skipped as blank" in p.stderr, p.stderr.strip()[-300:])
        for n, want in (("page", box[2:]), ("noborder", (W, H)), ("dark", (W, H))):
            o = os.path.join(out, n + ".png.avif")
            got = ispe(o) if os.path.exists(o) else None
            check(f"batch: {n} size", got == want, f"{got} (want {want})")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
