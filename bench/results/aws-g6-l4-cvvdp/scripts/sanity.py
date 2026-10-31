"""Before the run: (1) Pillow's JPEG decode (CVVDP's reference) equals libjxl's (ssimulacra2 of the JPEG against
Pillow's PNG with the same ICC profile must be 100); (2) CVVDP on the GPU: time and memory on the smallest and the
largest photo against a JPEG q90 re-encode, and identical inputs give 10."""
import io, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/av1gpu/bench"))
import bench
D = os.path.expanduser("~/images_gamut")
files = sorted(os.listdir(D), key=lambda f: os.path.getsize(os.path.join(D, f)))
for f in files[:3] + files[-3:]:
    im = Image.open(os.path.join(D, f))
    png = os.path.expanduser("~/work_sanity.png")
    kw = {"icc_profile": im.info["icc_profile"]} if im.info.get("icc_profile") else {}
    im.convert("RGB").save(png, **kw)
    p = bench.run([os.path.expanduser("~/opt/bin/ssimulacra2"), os.path.join(D, f), png])
    print("pillow vs libjxl decode", f, im.size, im.mode, "icc" if kw else "-", p.stdout.strip())
cv = bench.Cvvdp("standard_fhd")
print(cv.info)
for f in [files[0], max(files, key=lambda f: np.prod(Image.open(os.path.join(D, f)).size))]:
    src = os.path.join(D, f)
    b = io.BytesIO(); Image.open(src).convert("RGB").save(b, "JPEG", quality=90); b.seek(0)
    t = os.path.expanduser("~/work_sanity_q90.png"); Image.open(b).save(t)
    for rep in range(2):
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        j = cv(src, t, t)
        torch.cuda.synchronize()
        print(f, Image.open(src).size, "jod", j, f"{time.time() - t0:.2f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    ref = bench.srgb8(src)
    with cv.lock:
        print("identical", float(cv.metric.predict(ref, ref, dim_order="HWC")[0]))
