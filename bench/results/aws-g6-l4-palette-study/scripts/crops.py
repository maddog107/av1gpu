#!/usr/bin/env python3
"""Side-by-side crops at equal file size (run on the instance after study.py).

For each image below: the target size is libaom config A's q60 file (8-bit 4:2:0).  Configs C and D get the highest
avifenc -q whose file is no larger, av1gpu (default options) the lowest --qindex whose file is no larger.  Every file
is decoded with dav1d; each region (192 x 192 source pixels) is shown at 2x (nearest neighbour) as
  source | A | C (palette) | D (palette + IntraBC) | av1gpu main | av1gpu feat/palette (when $AV1GPU_PAL is set)
with the file size and the whole image's SSIMULACRA2 above each panel.  Output: ~/pal/crops/*.png, crops.json."""
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

H = os.path.expanduser("~")
PAL = f"{H}/pal"
AVIFENC, AVIFDEC, SSIM2 = f"{H}/opt/bin/avifenc", f"{H}/opt/bin/avifdec", f"{H}/opt/bin/ssimulacra2"
AV1GPU = f"{H}/av1gpu/build/av1gpu"
AV1GPU_PAL = os.environ.get("AV1GPU_PAL")   # the palette branch's binary (optional panel)
CFG = {"C": ["-a", "tune-content=screen", "-a", "enable-intrabc=0"], "D": ["-a", "tune-content=screen"]}
S = 192
JOBS = [  # (set, image, [(x, y) of each S x S region])
    ("comics", "comics_americasgreatest1941_p138.jpg", [(120, 1050), (1500, 1360)]),
    ("comics", "comics_superworld1940_p129.jpg", [(250, 380), (240, 1520)]),
    ("flat_png", "flat_wikipe_tan_mopping.png", [(860, 430), (1130, 80)]),
    ("flat_jpg", "flat_wikipe_tan_mopping.jpg", [(860, 430), (1130, 80)]),
    ("screen", "screen_synth_shot_text.png", [(80, 100), (900, 600)]),
    ("screen", "screen_libreoffice_writer_7_5_0_windows10.png", [(0, 40), (660, 740)]),
]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def aom(src, dst, q, cfg):
    run([AVIFENC, "-q", str(q), "-y", "420", "-j", "16", "--ignore-exif", "--ignore-xmp", "-s", "6", "-a", "tune=iq"] + CFG[cfg] +
        [src, dst])
    return os.path.getsize(dst)


def gpu(src, dst, qi, b=AV1GPU):
    run([b, src, dst, "--qindex", str(qi)])
    return os.path.getsize(dst)


def search(fn, lo, hi, target, better_high):
    """the best setting in [lo, hi] whose file is <= target (better_high: higher setting = bigger file)"""
    best = None
    while lo <= hi:
        m = (lo + hi) // 2
        if fn(m) <= target:
            best = m
            if better_high:
                lo = m + 1
            else:
                hi = m - 1
        else:
            if better_high:
                hi = m - 1
            else:
                lo = m + 1
    return best


def main():
    out = f"{PAL}/crops"
    os.makedirs(out, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    meta = []
    for st, img, regions in JOBS:
        src = f"{PAL}/sets/{st}/{img}"
        a = f"{PAL}/out/aom/420_A/{st}/{img}.q60.avif"
        target = os.path.getsize(a)
        files = {"A": (a, 60)}
        td = tempfile.mkdtemp(dir=out)
        for c in ("C", "D"):
            q = search(lambda q: aom(src, f"{td}/t.avif", q, c), 0, 100, target, True)
            aom(src, f"{td}/{c}.avif", q, c)
            files[c] = (f"{td}/{c}.avif", q)
        qi = search(lambda v: gpu(src, f"{td}/t.avif", v), 1, 255, target, False)
        gpu(src, f"{td}/gpu.avif", qi)
        files["av1gpu"] = (f"{td}/gpu.avif", qi)
        if AV1GPU_PAL:
            qp = search(lambda v: gpu(src, f"{td}/t.avif", v, AV1GPU_PAL), 1, 255, target, False)
            gpu(src, f"{td}/gpupal.avif", qp, AV1GPU_PAL)
            files["av1gpu_pal"] = (f"{td}/gpupal.avif", qp)
        dec, info = {}, {}
        for k, (f, q) in files.items():
            png = f"{td}/{k}.png"
            run([AVIFDEC, "-c", "dav1d", f, png])
            s2 = float(run([SSIM2, src, png]).stdout.split()[-1])
            dec[k] = Image.open(png).convert("RGB")
            info[k] = dict(bytes=os.path.getsize(f), setting=q, ssimulacra2=s2)
        srcim = Image.open(src).convert("RGB")
        meta.append(dict(set=st, image=img, target=target, files=info))
        for ri, (x, y) in enumerate(regions):
            panels = [("source", srcim, "")] + [(k, dec[k], f"{info[k]['bytes']:,} B  s2 {info[k]['ssimulacra2']:.1f}")
                                                for k in ("A", "C", "D", "av1gpu", "av1gpu_pal") if k in dec]
            W = 2 * S
            sheet = Image.new("RGB", (len(panels) * (W + 8), W + 44), (255, 255, 255))
            d = ImageDraw.Draw(sheet)
            for i, (name, im, lab) in enumerate(panels):
                crop = im.crop((x, y, x + S, y + S)).resize((W, W), Image.NEAREST)
                sheet.paste(crop, (i * (W + 8), 44))
                title = {"A": "libaom A (no palette)", "C": "libaom C (palette)", "D": "libaom D (palette + IntraBC)",
                         "av1gpu": "av1gpu main", "av1gpu_pal": "av1gpu palette", "source": "source"}[name]
                d.text((i * (W + 8) + 4, 2), title, font=font, fill=(0, 0, 0))
                d.text((i * (W + 8) + 4, 22), lab, font=font, fill=(60, 60, 60))
            sheet.save(f"{out}/{os.path.splitext(img)[0]}_{st}_r{ri + 1}.png", optimize=True)
        print(st, img, target, {k: (v["bytes"], v["setting"], round(v["ssimulacra2"], 1)) for k, v in info.items()}, flush=True)
    json.dump(meta, open(f"{out}/crops.json", "w"), indent=1)


if __name__ == "__main__":
    main()
