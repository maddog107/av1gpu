#!/usr/bin/env python3
"""--resize, --grayscale / --colorspace Gray, --auto-level / --level / --gamma / --negate.

On a PNG (and on PNGs made here): --resize gives ImageMagick's -resize sizes for every geometry form (computed here
with ImageMagick's rules, and asked of ImageMagick itself when it is installed), in single-image mode and on --batch list
lines (resize=); --resize 'WxH>' gives the bytes of --max-size WxH; an identity geometry gives the bytes of a plain
encode.  --grayscale writes a monochrome AVIF whose luma follows the method on gray and coloured pixels; Rec601Luma keeps
the planes' own luma.  The tone operators move known samples where ImageMagick's formulas put them (--dump-source, 4:4:4,
8 and 10 bits), --level with plain numbers is in Q16 units, and an identity --level / --gamma 1 gives the bytes of a
plain encode.  Bad values: status 2.  With ImageMagick (`magick`) installed, the --dump-source planes are also compared
with ImageMagick's own -resize / -grayscale / -level / -gamma output (mean difference below 0.6 of a step).  JPEGs, EXIF
rotations and 4:2:0 samples: imageops_jpeg_test.py.

  tests/imageops_test.py --bin build/av1gpu --image tests/gate_images/cfl_edge_211x149.png
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
    d = open(path, "rb").read()
    i = d.find(b"ispe")
    return struct.unpack(">II", d[i + 8:i + 16]) if i >= 0 else None


def mono_avif(path):
    """the av1C box's mono_chrome flag"""
    d = open(path, "rb").read()
    i = d.find(b"av1C")
    return bool(d[i + 6] & 0x10) if i >= 0 else None


def png_size(path):
    return struct.unpack(">II", open(path, "rb").read()[16:24])


def write_png(path, w, h, px):
    rows = [b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h)]

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def im_resize(w, h, g):
    """ImageMagick's -resize size for geometry g (its GetGeometry and ParseMetaGeometry; at least 1 here)"""
    rnd = lambda v: max(1, math.floor(v + 0.5))
    m = re.fullmatch(r"(@)?([\d.]*)(%?)(@)?(?:x([\d.]*)(%?))?([!<>^%]*)", g)
    a1, gw, p1, a2, gh, p2, fl = m.groups()
    pct = "%" in (p1 + (p2 or "") + fl)
    gw = float(gw) if gw else None
    gh = float(gh) if gh else None
    if a1 or a2:
        k = math.sqrt(gw / (w * h))
        tw, th = rnd(w * k), rnd(h * k)
        return (min(tw, w), min(th, h)) if ">" in fl else (tw, th)
    if pct:   # (a percentage is already the size that > and < compare with)
        sw = 100 if gw is None else gw
        sh = sw if gh is None else gh
        return rnd(w * sw / 100), rnd(h * sh / 100)
    gw = None if gw is None else rnd(gw)
    gh = None if gh is None else rnd(gh)
    if "!" in fl:
        tw, th = gw or w, gh or h
    elif gw and gh:
        k = (max if "^" in fl else min)(gw / w, gh / h)
        tw, th = rnd(w * k), rnd(h * k)
    elif "^" in fl:   # one number: the smaller side's
        n = gw or gh
        k = max(n / w, n / h) if gw else max(n / h, n / w)
        tw, th = rnd(w * k), rnd(h * k)
    elif gw:
        tw, th = gw, rnd(h * gw / w)
    else:
        tw, th = rnd(w * gh / h), gh
    if ">" in fl:
        tw, th = min(tw, w), min(th, h)
    if "<" in fl:
        tw, th = max(tw, w), max(th, h)
    return tw, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_ops_")
    fails = []
    magick = shutil.which("magick")

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    def run(*args):
        return subprocess.run([a.bin] + [str(x) for x in args], capture_output=True, text=True, timeout=300)

    def same(img, opts1, opts2):
        o1, o2 = os.path.join(t, "a.avif"), os.path.join(t, "b.avif")
        p1, p2 = run(img, o1, *opts1), run(img, o2, *opts2)
        return p1.returncode == 0 and p2.returncode == 0 and open(o1, "rb").read() == open(o2, "rb").read()

    def dump(img, *opts, depth=8):
        d = os.path.join(t, "d.yuv")
        p = run(img, os.path.join(t, "o.avif"), "--yuv", "444", "--dump-source", d, *(["--depth", "10"] if depth == 10 else []), *opts)
        if p.returncode != 0:
            return None
        raw = open(d, "rb").read()
        return list(struct.unpack("<%dH" % (len(raw) // 2), raw)) if depth == 10 else list(raw)

    try:
        img = os.path.join(t, "src.png")
        shutil.copy(a.image, img)
        W, H = png_size(img)
        # ---- --resize sizes ----
        for g in ("100x", "x100", "100x100", "100x100!", "100x100^", "100x100<", "300x300<", "300x300>", "50%", "50%x25%",
                  "150%", "@5000", "@100000", "7x", "100x50%", "100x100>!", "3x2!", "x50%", "50%x", "150%>", "50%<", "x150%",
                  "100!", "x100!", "100^", "x100^", "100x^", "@5000<", "@5000>", "5000@", "@2000000", "99.5x", "100.6x"):
            o = os.path.join(t, "r.avif")
            p = run(img, o, "--resize", g)
            got = ispe(o) if p.returncode == 0 else None
            want = im_resize(W, H, g)
            if magick:
                r = subprocess.run([magick, img, "-resize", g, "-format", "%w %h", "info:"], capture_output=True, text=True)
                if r.returncode == 0:
                    want = tuple(int(v) for v in r.stdout.split())
            check(f"--resize {g}", got == want, f"{got} (want {want}) {p.stderr.strip()[-200:]}")
        check("--resize 'WxH>' == --max-size WxH", same(img, ["--resize", "100x100>"], ["--max-size", "100x100"]))
        check("--resize 100% == plain encode", same(img, ["--resize", "100%"], []))
        check("the last of --max-size / --resize counts", same(img, ["--max-size", "50x50", "--resize", "150%"], ["--resize", "150%"]))
        # ---- batch list resize= ----
        src = os.path.join(t, "in")
        os.makedirs(src)
        lines = [("a.png\tresize=50%", "50%"), ("b.png\tresize=120x80!", "120x80!"), ("c.png\tresize=10x10+1+1", None),
                 ("d.png\tmax-size=x30\tresize=@2000", "@2000")]
        for l, _ in lines:
            shutil.copy(img, os.path.join(src, l.split("\t")[0]))
        lst = os.path.join(t, "list.txt")
        open(lst, "w").write("".join(os.path.join(src, l) + "\n" for l, _ in lines))
        out = os.path.join(t, "out")
        p = run("--batch", lst, "--outdir", out, "--resize", "x40")
        check("batch: exit 0", p.returncode == 0, p.stderr.strip()[-300:])
        for l, g in lines:
            o = os.path.join(out, l.split("\t")[0] + ".avif")
            if g is None:
                check(f"batch: {l!r} skipped", not os.path.exists(o) and "no offset" in p.stderr)
                continue
            got = ispe(o) if os.path.exists(o) else None
            check(f"batch: {l!r}", got == im_resize(W, H, g), f"{got} (want {im_resize(W, H, g)})")
        # ---- --grayscale ----
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (40, 120, 200), (128, 128, 128), (250, 240, 10)]
        sw = os.path.join(t, "swatch.png")
        write_png(sw, 8 * len(colours), 8, lambda x, y: colours[x // 8])
        methods = {"Rec601Luma": lambda r, g, b: 0.298839 * r + 0.586811 * g + 0.114350 * b,
                   "Rec709Luma": lambda r, g, b: 0.212656 * r + 0.715158 * g + 0.072186 * b,
                   "Average": lambda r, g, b: (r + g + b) / 3, "Brightness": lambda r, g, b: max(r, g, b),
                   "Lightness": lambda r, g, b: (max(r, g, b) + min(r, g, b)) / 2,
                   "RMS": lambda r, g, b: math.sqrt((r * r + g * g + b * b) / 3)}
        for m, f in methods.items():
            o = os.path.join(t, "g.avif")
            p = run(sw, o, "--grayscale", m)
            d = dump(sw, "--grayscale", m)
            ok = p.returncode == 0 and mono_avif(o) and d is not None and len(d) == 8 * len(colours) * 8
            err = max(abs(d[8 * i + 3 + 3 * 8 * len(colours)] - f(*c)) for i, c in enumerate(colours)) if ok else -1
            check(f"--grayscale {m}: monochrome, gray of each colour", ok and err <= 2, f"max error {err:.2f}")
        check("--colorspace Gray == --grayscale Rec709Luma", same(sw, ["--colorspace", "Gray"], ["--grayscale", "rec709luma"]))
        # ---- tone operators on known samples ----
        ramp = os.path.join(t, "ramp.png")
        write_png(ramp, 64, 4, lambda x, y: (4 * x, 4 * x, 4 * x))
        c = lambda v: min(1.0, max(0.0, v))
        for opts, f in ((["--level", "10%,90%"], lambda v: (v - 0.1) / 0.8), (["--level", "20%,80%,2"], lambda v: c((v - 0.2) / 0.6) ** 0.5),
                        (["--level", "10%"], lambda v: (v - 0.1) / 0.8), (["--level", "6553.5,58981.5"], lambda v: (v - 0.1) / 0.8),
                        (["--gamma", "2.2"], lambda v: v ** (1 / 2.2)), (["--negate"], lambda v: 1 - v),
                        (["--auto-level"], lambda v: v / (252 / 255)),
                        (["--negate", "--gamma", "2", "--level", "5%,95%"], lambda v: 1 - c((v - 0.05) / 0.9) ** 0.5)):
            for depth in (8, 10):
                d = dump(ramp, *opts, depth=depth)
                top = 1023 if depth == 10 else 255
                ok = d is not None
                err = max(abs(d[x] - top * min(1, max(0, f(min(1, max(0, 4 * x / 255)))))) for x in range(64)) if ok else -1
                # (10-bit: the 8-bit source's rounding, magnified where a gamma curve is steep)
                check(f"{' '.join(opts)} ({depth}-bit)", ok and err <= (1.5 if depth == 8 else 5), f"max error {err:.2f}")
        check("--level 0%,100% == plain encode", same(img, ["--level", "0%,100%"], []))
        check("--level: one % makes both percentages", same(img, ["--level", "10,90%"], ["--level", "10%,90%"]))
        check("--colorspace in any case", same(sw, ["--colorspace", "GRAY"], ["--colorspace", "gray"]))
        check("--gamma 1 == plain encode", same(img, ["--gamma", "1"], []))
        # 4:2:0: a gray ramp stays gray (chroma at the middle)
        o = os.path.join(t, "t.avif")
        d420 = os.path.join(t, "d420.yuv")
        p = run(img, o, "--level", "10%,90%", "--dump-source", d420)
        check("4:2:0 --level", p.returncode == 0 and os.path.getsize(d420) == W * H + 2 * ((W + 1) // 2) * ((H + 1) // 2))
        # ---- errors ----
        for opts, word in ((["--resize", "abc"], "geometry"), (["--resize", "10x10+1+1"], "no offset"), (["--resize", "@50%"], "AREA"),
                           (["--grayscale", "luma"], "Rec601Luma"), (["--colorspace", "CMYK"], "Gray"), (["--level", "90%,10%"], "above"),
                           (["--level", "a"], "BLACK"), (["--gamma", "0"], "above 0"), (["--gamma", "101"], "up to 100"),
                           (["--resize", "0.4x"], "rounds to 0"), (["--resize", "10x0.4!"], "rounds to 0"), (["--resize", "@5@"], "geometry")):
            p = run(img, os.path.join(t, "e.avif"), *opts)
            check(f"bad {' '.join(opts)}: status 2", p.returncode == 2 and word in p.stderr, (p.stderr.strip().splitlines() or [""])[0][-200:])
        # ---- ImageMagick ----
        if magick:
            for imops, opts in (("-resize 150%", ["--resize", "150%"]), ("-resize 60x60^", ["--resize", "60x60^"]),
                                ("-colorspace Gray", ["--colorspace", "Gray"]), ("-grayscale Average", ["--grayscale", "Average"]),
                                ("-level 10%,90%", ["--level", "10%,90%"]), ("-gamma 0.8", ["--gamma", "0.8"]), ("-negate", ["--negate"])):
                ref = os.path.join(t, "ref.rgb")
                subprocess.run([magick, img, *imops.split(), "-depth", "8", "-colorspace", "sRGB", "-type", "TrueColor", "rgb:" + ref], check=True)
                d = dump(img, *opts)
                rgb = open(ref, "rb").read()
                n = len(rgb) // 3
                ys = [round(0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2]) for i in range(n)]
                mad = sum(abs(d[i] - ys[i]) for i in range(n)) / n if d and len(d) >= n else 99
                check(f"ImageMagick {imops}: luma", mad < 0.6, f"mean difference {mad:.3f}")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
