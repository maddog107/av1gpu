"""Diagnostic: CVVDP on luma only (both images converted to 8-bit grey by Pillow after the sRGB conversion, so CVVDP sees
no colour), for some codecs' files, to see whether a JOD gap is in luma or in chroma.
cvvdp_luma.py OUT.json CODEC=DIR[,DIR...] ..."""
import concurrent.futures as cf, json, os, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/av1gpu/bench"))
import bench
SRC = os.path.expanduser("~/images_gamut"); AVIFDEC = os.path.expanduser("~/opt/bin/avifdec")
cv = bench.Cvvdp("standard_fhd")
inv = {bench.qindex_for_quality(x): x for x in (30, 40, 50, 60, 70, 80, 85, 90)}
jobs = []
for arg in sys.argv[2:]:
    codec, dirs = arg.split("=")
    for d in dirs.split(","):
        for f in sorted(os.listdir(d)):
            if f.endswith(".avif"):
                img, q = f.rsplit(".q", 1); q = int(q[:-5])
                jobs.append((codec, img, inv[q] if codec == "av1gpu" else q, os.path.join(d, f)))


def grey(a):
    return np.asarray(Image.fromarray(a).convert("L"))


def one(job):
    codec, img, q, path = job
    with tempfile.TemporaryDirectory() as td:
        png = os.path.join(td, "d.png")
        bench.run([AVIFDEC, "-c", "dav1d", path, png])
        ref, test = grey(bench.srgb8(os.path.join(SRC, img))), grey(bench.srgb8(png))
        with cv.lock:
            j = float(cv.metric.predict(test, ref, dim_order="HW")[0])
    return dict(codec=codec, image=img, q=q, bytes=os.path.getsize(path), cvvdp_luma=j)


with cf.ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(one, jobs))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(len(rows), "rows")
