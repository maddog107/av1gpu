#!/usr/bin/env python3
"""--trim's kept size against ImageMagick's -trim box (`magick IMG -fuzz F -format %@ info:`), image by image.

One av1gpu --batch call over DIR with `--trim --fuzz F` (4:4:4, so chroma subsampling does not widen boxes; turbo3,
since only the sizes are read), then ImageMagick on every image (32 at a time).  An image ImageMagick finds all border
(0x0) is encoded whole by av1gpu.  Prints the counts and every image whose sizes differ, with both boxes' sizes.

  trim_vs_imagemagick.py --bin build/av1gpu --fuzz 10% DIR [DIR ...]
"""
import argparse, os, re, struct, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

def ispe(p):
    d = open(p, "rb").read(); i = d.find(b"ispe")
    return struct.unpack(">II", d[i + 8:i + 16])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--fuzz", default="10%"); ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    imgs = sorted(os.path.join(d, n) for d in a.dirs for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    t = tempfile.mkdtemp()
    lst = os.path.join(t, "list")
    with open(lst, "w") as f:
        for i, p in enumerate(imgs):
            link = os.path.join(t, "%06d%s" % (i, os.path.splitext(p)[1])); os.symlink(os.path.abspath(p), link); f.write(link + "\n")
    out = os.path.join(t, "out")
    r = subprocess.run([a.bin, "--batch", lst, "--outdir", out, "--yuv", "444", "--preset", "turbo3", "--max-mp", "100",
                        "--max-input-mp", "100", "--trim", "--fuzz", a.fuzz], capture_output=True, text=True)
    print("\n".join(l for l in r.stderr.splitlines() if l.startswith("[av1gpu]") and "stage ms" not in l))
    def im(p):
        s = subprocess.run(["magick", p, "-fuzz", a.fuzz, "-format", "%@ %w %h", "info:"], capture_output=True, text=True).stdout.split()
        bw, bh = (int(v) for v in re.match(r"(\d+)x(\d+)", s[0]).groups())
        return (int(s[1]), int(s[2])) if bw == 0 else (bw, bh)
    with ThreadPoolExecutor(32) as ex:
        want = list(ex.map(im, imgs))
    same = 0; diff = []
    for i, (p, w) in enumerate(zip(imgs, want)):
        o = os.path.join(out, "%06d%s.avif" % (i, os.path.splitext(p)[1]))
        g = ispe(o) if os.path.exists(o) else None
        if g == w: same += 1
        else: diff.append((os.path.basename(p), w, g))
    print("fuzz %s: %d images, %d of the same size as ImageMagick's -trim, %d differ" % (a.fuzz, len(imgs), same, len(diff)))
    for d in diff: print("  %s: ImageMagick %s, av1gpu %s" % d)

main()
