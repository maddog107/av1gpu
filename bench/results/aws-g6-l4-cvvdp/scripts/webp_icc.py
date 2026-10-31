"""Side check: SSIMULACRA2 of the WebP files with the .webp's own ICC profile attached to dwebp's PNG (dwebp drops it,
so the published WebP scores compare an untagged PNG with the ICC-tagged source).  Only images whose WebP has a profile.
webp_icc.py WEBPDIR [WEBPDIR...] OUT.json"""
import concurrent.futures as cf, json, os, subprocess, sys, tempfile
from PIL import Image
SRC = os.path.expanduser("~/images_gamut"); SSIM2 = os.path.expanduser("~/opt/bin/ssimulacra2")
dirs, out = sys.argv[1:-1], sys.argv[-1]
jobs = [os.path.join(d, f) for d in dirs for f in sorted(os.listdir(d)) if f.endswith(".webp")]


def one(path):
    icc = Image.open(path).info.get("icc_profile")
    if not icc:
        return None
    img, q = os.path.basename(path).rsplit(".q", 1)
    with tempfile.TemporaryDirectory() as td:
        png, png2 = os.path.join(td, "d.png"), os.path.join(td, "i.png")
        subprocess.run(["dwebp", "-quiet", path, "-o", png], check=True)
        Image.open(png).save(png2, icc_profile=icc)
        p = subprocess.run([SSIM2, os.path.join(SRC, img), png2], capture_output=True, text=True)
        return dict(image=img, q=int(q[:-5]), ssimulacra2_icc=float(p.stdout.split()[-1]))


with cf.ThreadPoolExecutor(16) as ex:
    rows = [r for r in ex.map(one, jobs) if r]
json.dump(rows, open(out, "w"), indent=1)
print(len(rows), "rows")
