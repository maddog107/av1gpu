#!/usr/bin/env python3
"""match.py: the branch's crop / resize against ImageMagick 7 (~/im7/AppRun, Q16 HDRI).  For each case the encoder's
input planes (--dump-source) are compared with ImageMagick's result converted by av1gpu's own PNG path the same way
(the same options, no crop / resize): mean absolute difference (in steps of the output depth) and PSNR per plane,
and the output size against ImageMagick's geometry.  RGBA: ImageMagick with -filter Lanczos (its default for images
with alpha is Mitchell), colour compared where the alpha is visible (ImageMagick zeroes fully transparent colour).
~/results/match.jsonl and ~/results/match.md."""
import glob, json, os, struct, subprocess
import numpy as np
from PIL import Image

H = os.path.expanduser("~")
B = f"{H}/cr/build/av1gpu"
IM = f"{H}/im7/AppRun"
W = f"{H}/work/match"
os.makedirs(W, exist_ok=True)
R = f"{H}/results"


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{cmd}: {p.stderr[-500:]}")
    return p


def ispe(path):
    d = open(path, "rb").read()
    i = d.find(b"ispe")
    return struct.unpack(">II", d[i + 8:i + 16])


def irot(path):
    d = open(path, "rb").read()
    i = d.find(b"irot")
    return d[i + 4] & 3 if i >= 0 else 0


def planes(path, w, h, fmt, depth, alpha):
    d = np.fromfile(path, dtype="<u2" if depth == 10 else "u1").astype(np.float64)
    cw, ch = ((w + 1) // 2, (h + 1) // 2) if fmt == "420" else (w, h)
    out, at = {}, 0
    for name, (pw, ph) in (("Y", (w, h)), ("U", (cw, ch)), ("V", (cw, ch))) + ((("A", (w, h)),) if alpha else ()):
        out[name] = d[at:at + pw * ph].reshape(ph, pw)
        at += pw * ph
    assert at == d.size, (path, at, d.size)
    return out


def case(name, src, opts, im_ops, fmt, depth, alpha=False, im_filter=None):
    a = f"{W}/{name}"
    run([B, src, a + ".avif", "--dump-source", a + ".dump"] + opts)
    w, h = ispe(a + ".avif")
    png = f"{W}/{name}_im.png"
    run([IM, src] + (["-filter", im_filter] if im_filter else []) + im_ops + ["-strip", "-depth", "16" if depth == 10 else "8", png])
    iw, ih = Image.open(png).size
    base = [o for o in opts if o in ("--yuv", "444", "420", "--depth", "10", "8", "--lossless", "--ycgco-re")]
    run([B, png, f"{W}/{name}_im.avif", "--dump-source", f"{W}/{name}_im.dump"] + base)
    r = dict(case=name, src=os.path.basename(src), opts=" ".join(opts), im=" ".join(im_ops), size=f"{w}x{h}", im_size=f"{iw}x{ih}",
             size_ok=(w, h) == (iw, ih), fmt=fmt, depth=depth, irot=irot(a + ".avif"))
    if (w, h) == (iw, ih):
        p, q = planes(a + ".dump", w, h, fmt, depth, alpha), planes(f"{W}/{name}_im.dump", w, h, fmt, depth, alpha)
        peak = 1023.0 if depth == 10 else 255.0
        mask = {}
        if alpha:   # colour where ImageMagick's alpha is visible (4:2:0 chroma: all of its 2x2, as av1gpu's 4:2:0 conversion
            # of ImageMagick's PNG averages its zeroed colour under transparent pixels into chroma samples on the edge)
            vis = q["A"] > 0
            mask["Y"] = vis
            if fmt == "420":
                v = np.pad(vis, ((0, vis.shape[0] % 2), (0, vis.shape[1] % 2)), mode="edge")
                c = v[0::2, 0::2] & v[1::2, 0::2] & v[0::2, 1::2] & v[1::2, 1::2]
            else:
                c = vis
            mask["U"] = mask["V"] = c
        for k in p:
            e = (p[k] - q[k])[mask[k]] if k in mask else (p[k] - q[k]).ravel()
            mse = float(np.mean(e * e)) if e.size else 0.0
            r[k] = dict(mad=round(float(np.mean(np.abs(e))), 3), psnr=round(10 * np.log10(peak * peak / mse), 2) if mse > 0 else 99.0,
                        maxdiff=int(np.max(np.abs(e))) if e.size else 0, n=int(e.size))
    print(json.dumps(r), flush=True)
    with open(f"{R}/match.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    return r


open(f"{R}/match.jsonl", "w").close()
photos = sorted(glob.glob(f"{H}/images/*.jpg"), key=lambda f: Image.open(f).size[0] * Image.open(f).size[1])
big = photos[-1]                       # the largest size-range photo
mid = next(f for f in photos if Image.open(f).size[0] * Image.open(f).size[1] > 8e6)
bw, bh = Image.open(big).size
# a comic-like page: 3840x5586 4:4:4 JPEG (q94) made from the largest photo, as pickae's scans
page = f"{W}/page_3840x5586.jpg"
if not os.path.exists(page):
    run([IM, big, "-resize", "3840x5586!", "-sampling-factor", "1x1", "-quality", "94", page])
# EXIF orientation 6 (stored landscape, displayed portrait)
rot = f"{W}/rot6.jpg"
run(["cp", mid, rot])
run(["exiftool", "-q", "-m", "-all=", "-overwrite_original", rot])
run(["exiftool", "-q", "-m", "-n", "-Orientation=6", "-overwrite_original", rot])
kodak = f"{H}/ll/kodak/kodim23.png"
p16 = f"{H}/ll/png16_10/k2_10in16.png"   # a Kodak photo, 10 significant bits in 16
# the largest photo as a 16-bit RGB PNG: both sides start from the same RGB samples (no JPEG decode or 8-bit difference)
photo16 = f"{W}/photo16.png"
if not os.path.exists(photo16):
    run([IM, big, "-depth", "16", photo16])
rgba = max((f for f in glob.glob(f"{H}/data/rgba/set/*.png") if Image.open(f).mode in ("RGBA", "LA")),
           key=lambda f: Image.open(f).size[0] * Image.open(f).size[1])
rw, rh = Image.open(rgba).size
res = [
    case("crop_only_444_10", big, ["--crop", "1500x1000+333+777", "--yuv", "444", "--depth", "10"],
         ["-crop", "1500x1000+333+777", "+repage"], "444", 10),
    case("resize_only_444_10", big, ["--max-height", "2000", "--yuv", "444", "--depth", "10"], ["-resize", "x2000>"], "444", 10),
    case("page_crop_only_444_10", page, ["--crop", "3000x4000+333+777", "--yuv", "444", "--depth", "10"],
         ["-crop", "3000x4000+333+777", "+repage"], "444", 10),
    case("page_max_height_2960_444_10", page, ["--max-height", "2960", "--yuv", "444", "--depth", "10"], ["-resize", "x2960>"], "444", 10),
    case("photo_png16_resize_444_10", photo16, ["--max-height", "2000", "--yuv", "444", "--depth", "10"], ["-resize", "x2000>"], "444", 10),
    case("photo_png16_odd_crop_resize_420_10", photo16, ["--crop", "2001x1501+333+777", "--max-size", "1000x1000", "--depth", "10"],
         ["-crop", "2001x1501+333+777", "+repage", "-resize", "1000x1000>"], "420", 10),
    case("odd_crop_resize_420_8", big, ["--crop", "2001x1501+333+777", "--max-size", "1000x1000"],
         ["-crop", "2001x1501+333+777", "+repage", "-resize", "1000x1000>"], "420", 8),
    case("odd_crop_resize_420_10", big, ["--crop", "2001x1501+333+777", "--max-size", "1000x1000", "--depth", "10"],
         ["-crop", "2001x1501+333+777", "+repage", "-resize", "1000x1000>"], "420", 10),
    case("png8_resize_420_8", kodak, ["--max-size", "500x500"], ["-resize", "500x500>"], "420", 8),
    case("png8_odd_crop_resize_420_8", kodak, ["--crop", "601x401+51+37", "--max-size", "333x333"],
         ["-crop", "601x401+51+37", "+repage", "-resize", "333x333>"], "420", 8),
    case("png16_crop_resize_444_10", p16, ["--crop", "601x401+3+5", "--max-size", "300x300", "--yuv", "444", "--depth", "10"],
         ["-crop", "601x401+3+5", "+repage", "-resize", "300x300>"], "444", 10),
    case("rgba_resize_444_8", rgba, ["--max-size", f"{rw // 2}x{rh // 2}", "--yuv", "444"], ["-resize", f"{rw // 2}x{rh // 2}>"], "444", 8,
         alpha=True, im_filter="Lanczos"),
    case("rgba_crop_resize_420_8", rgba, ["--crop", f"{rw - 3}x{rh - 5}+1+3", "--max-size", f"{rw // 3}x{rh // 3}"],
         ["-crop", f"{rw - 3}x{rh - 5}+1+3", "+repage", "-resize", f"{rw // 3}x{rh // 3}>"], "420", 8, alpha=True, im_filter="Lanczos"),
    # displayed box 1600x1000 on a portrait-displayed (orientation 6) landscape-stored JPEG: stored box 1000x1600
    case("exif6_resize_420_8", rot, ["--max-size", "1600x1000"], ["-resize", "1000x1600>"], "420", 8),
]
with open(f"{R}/match.md", "w") as f:
    f.write("| case | source | output | ImageMagick | Y MAD / PSNR | U MAD / PSNR | V MAD / PSNR | alpha MAD / PSNR |\n|---|---|---|---|---:|---:|---:|---:|\n")
    for r in res:
        cell = lambda k: f"{r[k]['mad']} / {r[k]['psnr']} dB" if k in r else ""
        f.write(f"| {r['case']} | {r['src']} | {r['size']} | {r['im_size']}{'' if r['size_ok'] else ' (DIFFERENT)'} | "
                f"{cell('Y')} | {cell('U')} | {cell('V')} | {cell('A')} |\n")
print("MATCH_DONE")
