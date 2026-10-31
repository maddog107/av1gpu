#!/usr/bin/env python3
"""av1gpu benchmark: throughput and compression efficiency against libaom (avifenc) and WebP.

  bench/fetch_images.py                       # 100 CC BY 2.0 JPEGs -> bench/images
  bench/bench.py --images bench/images --out bench/results/my-gpu

Speed: `av1gpu --batch` over the whole set at the default settings (one warm-up run, then the median of
--repeats runs, from av1gpu's own --stats line), and the whole machine running one single-threaded avifenc
per CPU (the fastest way to use libaom on many images) at speed 6 and 9 (--aom-speeds).  Both run the set
repeated up to --min-mp megapixels, so that short sets measure steady-state throughput.

Quality: every codec encodes every image at several quality points; outputs are decoded to PNG (avifdec,
dwebp) and scored against the source JPEG with SSIMULACRA2 (libjxl's `ssimulacra2` tool).  For each image
the BD-rate (bytes at equal SSIMULACRA2, Bjontegaard with monotone cubic interpolation of log-size) is
computed against libaom speed 6; the table reports the median and mean over images.  Negative = smaller.

To time other av1gpu settings, pass a wrapper script as --av1gpu that adds the options, and --gpu-label.
--avifenc-threads N runs avifenc -j N instead, with cpu_count // N encodes at once (speed and quality phases).
avifenc 1.4 tiles automatically unless told otherwise (libavif: up to 8 tiles of at least 512 x 512 pixels,
whatever -j is); --no-tiling passes --tilerowslog2 0 --tilecolslog2 0 for one tile per image.
--yuv 444 and --depth 10 encode every AV1 codec in that format (avifenc -y 444 / -d 10, av1gpu --yuv 444 /
--depth 10); 10-bit outputs are decoded to 16-bit PNGs.
bench/formats.py compares the four formats.

Requires: avifenc, avifdec, cwebp, dwebp, ssimulacra2 on PATH (or the --avifenc, ... options), numpy, Pillow.
  Ubuntu:      apt install libavif-bin webp python3-numpy python3-pil
  ssimulacra2: from libjxl's dev tools, e.g.
                 git clone --depth 1 -b v0.11.1 https://github.com/libjxl/libjxl && cd libjxl
                 apt install libhwy-dev libbrotli-dev liblcms2-dev
                 cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON \\
                   -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON -DJPEGXL_FORCE_SYSTEM_LCMS2=ON \\
                   -DJPEGXL_ENABLE_SKCMS=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_SJPEG=OFF
                 cmake --build build --target ssimulacra2     # -> build/tools/ssimulacra2
  avifenc's libaom must be 3.12 or newer for tune=iq (the comparison av1gpu was tuned against); with an
  older libaom the script falls back to libaom's default tuning and says so in the summary.
--skip-gpu runs only the CPU codecs (useful for checking the setup on a machine without a GPU).
Writes OUT/results.json (every measurement) and OUT/summary.md.  WebP files with an ICC profile need Pillow: dwebp
leaves the profile out of its PNG, so the script copies it back before scoring (runs before 2026-10-31 scored
those 13 of the size-range photos as colour-shifted; see bench/results/aws-g6-l4-cvvdp).

CVVDP (optional, off by default): --cvvdp DISPLAY also scores every quality-phase file with ColorVideoVDP
(https://github.com/gfxdisp/ColorVideoVDP, `pip install cvvdp`, PyTorch; on the GPU when torch sees one) for one
of its display models, e.g. standard_fhd (24-inch 1920 x 1080, 200 cd/m^2, seen from 0.6 m under 250 lux, about
38 pixels per degree).  It uses the same decoded PNG as SSIMULACRA2, whose scoring does not change.  Both images
go in as 8-bit sRGB: the source JPEG decoded by Pillow and the decoded output, each converted with LittleCMS
(relative colorimetric) from its own ICC profile if it has one, as SSIMULACRA2 does.  WebP takes the profile from
the .webp file, because dwebp leaves it out of its PNG.  CVVDP maps image pixels 1:1 to display pixels; it does
not scale an image to fit the screen.  The JOD scores (10 = no visible difference) give a second BD-rate table,
with JOD as the quality axis and --cvvdp-min-overlap (0.05 JOD, about what 2 SSIMULACRA2 points are on libaom's
curves) instead of 2 SSIMULACRA2 points as the least overlap.
They also give, for --cvvdp-target (9.5 JOD by default), how many images reach that JOD within the tested
qualities and the bits per pixel at which each one does (interpolated along its curve as in the BD-rate).
--avifdec-codec dav1d decodes with avifdec -c dav1d (default: avifdec's own choice).
  pip install cvvdp            # CUDA PyTorch from PyPI on Linux x86-64
  bench/bench.py --images bench/images_gamut --out OUT --skip-speed --cvvdp standard_fhd
On AWS's Deep Learning Base AMIs, LD_LIBRARY_PATH holds CUDA's own cuDNN, which breaks the one PyTorch ships: put
torch's (.../site-packages/nvidia/cudnn/lib) first.  The run in bench/results/aws-g6-l4-cvvdp/ has the commands.
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTS = (".jpg", ".jpeg")

# avifenc -q (tune=iq) -> qindex; the same table as av1f_qindex_for_quality() in av1f/include/av1f/tuneiq.h
_QP = [63, 63, 63, 62, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 54, 54, 53, 53, 52, 52,
       51, 51, 50, 50, 49, 49, 48, 48, 47, 46, 46, 45, 45, 44, 43, 43, 42, 42, 41, 40, 40, 39, 39, 38, 37, 37,
       36, 36, 35, 34, 33, 33, 32, 31, 30, 30, 29, 28, 27, 27, 26, 25, 24, 24, 23, 22, 21, 21, 20, 19, 18, 18,
       17, 16, 15, 15, 14, 13, 12, 12, 11, 10, 9, 9, 8, 7, 6, 6, 5, 4, 3, 3, 2, 1, 0]


def qindex_for_quality(q):
    p = _QP[max(0, min(100, q))]
    return 255 if p == 63 else 249 if p == 62 else max(1, 4 * p)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def tool_version(cmd):
    try:
        p = run(cmd)
        return (p.stdout + p.stderr).strip().splitlines()[0][:200]
    except (OSError, IndexError):
        return "unavailable"


def environment(a):
    env = dict(platform=platform.platform(), python=platform.python_version(), cpus=os.cpu_count())
    try:
        env["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        env["cpu"] = platform.processor()
    p = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"]) \
        if shutil.which("nvidia-smi") else None
    env["gpu"] = p.stdout.strip() if p and p.returncode == 0 else "none"
    env["avifenc"] = "avifenc " + tool_version([a.avifenc, "--version"]).removeprefix("Version: ")
    env["cwebp"] = tool_version([a.cwebp, "-version"])
    return env


# ------------------------------------------------------------------------------------------- BD-rate
def _pchip_slopes(x, y):
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    if len(x) == 2:
        m[:] = d[0]
        return m
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    # end points: one-sided three-point estimate, shape preserving (as scipy's PchipInterpolator)
    for k, h0, h1, d0, d1 in ((0, h[0], h[1], d[0], d[1]), (-1, h[-1], h[-2], d[-1], d[-2])):
        e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(e) != np.sign(d0):
            e = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(e) > abs(3 * d0):
            e = 3 * d0
        m[k] = e
    return m


def _pchip_eval(x, y, xs):
    m = _pchip_slopes(x, y)
    i = np.clip(np.searchsorted(x, xs) - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (xs - x[i]) / h
    return ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * h * m[i] +
            (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * h * m[i + 1])


def _curve(points):
    """(bytes, score) points -> quality-sorted arrays with strictly increasing quality"""
    pts = sorted((s, np.log(b)) for b, s in points if b > 0 and s is not None and np.isfinite(s))
    xs, ys = [], []
    for s, lb in pts:
        if xs and s <= xs[-1]:
            continue
        xs.append(s)
        ys.append(lb)
    return np.array(xs), np.array(ys)


def bd_rate(test, anchor, min_overlap=2.0):
    """Bjontegaard rate difference (fraction) of test vs anchor; None if the curves barely overlap"""
    xt, yt = _curve(test)
    xa, ya = _curve(anchor)
    if len(xt) < 2 or len(xa) < 2:
        return None
    lo, hi = max(xt[0], xa[0]), min(xt[-1], xa[-1])
    if hi - lo < min_overlap:
        return None
    xs = np.linspace(lo, hi, 256)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    diff = trapezoid(_pchip_eval(xt, yt, xs) - _pchip_eval(xa, ya, xs), xs) / (hi - lo)
    return float(np.expm1(diff))


def bd_table(quality, sizes, metric="ssimulacra2", min_overlap=2.0):
    """per-image BD-rates on `metric` of every codec against libaom speed 6 and against WebP:
    ({"<codec>_vs_<anchor>": {n, median, mean}}, {"<codec>_vs_aom_s6": {size class: {n, median}}})"""
    per = {}
    for codec, rows in quality.items():
        for r in rows:
            per.setdefault(codec, {}).setdefault(r["image"], []).append((r["bytes"], r.get(metric)))
    bd, by_size = {}, {}
    for anchor in ("aom_s6", "webp"):
        if anchor not in per:
            continue
        for codec in per:
            if codec == anchor:
                continue
            vals = [bd_rate(per[codec][im], per[anchor][im], min_overlap) for im in per[codec] if im in per[anchor]]
            vals = [v for v in vals if v is not None]
            if vals:
                bd[f"{codec}_vs_{anchor}"] = dict(n=len(vals), median=statistics.median(vals), mean=statistics.fmean(vals))
            if anchor != "aom_s6":
                continue
            for cls, lo, hi in SIZE_CLASSES:
                v = [bd_rate(per[codec][im], per[anchor][im], min_overlap) for im in per[codec]
                     if im in per[anchor] and lo <= sizes[im] < hi]
                v = [x for x in v if x is not None]
                if v:
                    by_size.setdefault(f"{codec}_vs_{anchor}", {})[cls] = dict(n=len(v), median=statistics.median(v))
    return bd, by_size


# ------------------------------------------------------------------------------------------- encoders
def avifenc_threads(a):
    return getattr(a, "avifenc_threads", 1)


def avifenc_format(a):
    """avifenc chroma format and bit depth options (default: -y 420, 8 bits)"""
    d = getattr(a, "depth", 8)
    return ["-y", str(getattr(a, "yuv", "420"))] + (["-d", str(d)] if d != 8 else [])


def gpu_format(a):
    """av1gpu chroma format and bit depth options (none for 8-bit 4:2:0)"""
    y, d = str(getattr(a, "yuv", "420")), getattr(a, "depth", 8)
    return (["--yuv", y] if y != "420" else []) + (["--depth", str(d)] if d != 8 else [])


def format_label(a):
    """'' for 8-bit 4:2:0, else e.g. ', 10-bit 4:4:4'"""
    y, d = str(getattr(a, "yuv", "420")), getattr(a, "depth", 8)
    return "" if (y, d) == ("420", 8) else f", {d}-bit {y[0]}:{y[1]}:{y[2]}"


def avifenc_parallel(a, cpus):
    """how many avifenc -j N run at once on `cpus` CPUs"""
    return max(1, cpus // avifenc_threads(a))


def avifenc_cmd(a, src, dst, q, speed, tune_iq, codec="aom"):
    """codec "aom": libaom at speed; "svt": SVT-AV1 at preset `speed` (avifenc -c svt, -a preset= so 11-13 work)."""
    cmd = [a.avifenc, "-q", str(q)] + avifenc_format(a) + ["-j", str(avifenc_threads(a)), "--ignore-exif", "--ignore-xmp"]
    if getattr(a, "no_tiling", False):
        cmd += ["--tilerowslog2", "0", "--tilecolslog2", "0"]
    if codec == "svt":
        cmd += ["-c", "svt", "-a", f"preset={speed}"] + (["-a", f"tune={a.svt_tune}"] if a.svt_tune else [])
    else:
        cmd += ["-s", str(speed)] + (["-a", "tune=iq"] if tune_iq else [])
    return cmd + [src, dst]


def avifenc_supports_tune_iq(a, src, work):
    p = run(avifenc_cmd(a, src, os.path.join(work, "probe.avif"), 60, 9, True))
    return p.returncode == 0


def decode_png(a, path, png):
    if path.endswith(".webp"):
        if run([a.dwebp, path, "-o", png]).returncode != 0:
            return False
        icc = file_icc(path)
        if icc:  # dwebp leaves out the ICC profile cwebp stored; put it back so the PNG is scored in its own colours
            from PIL import Image
            im = Image.open(png)
            im.load()
            im.save(png, icc_profile=icc)
        return True
    codec = getattr(a, "avifdec_codec", "")
    depth = ["-d", "16"] if getattr(a, "depth", 8) > 8 else []
    return run([a.avifdec] + (["-c", codec] if codec else []) + depth + [path, png]).returncode == 0


def score(a, src, path, work, cvvdp=None):
    """SSIMULACRA2 of `path`, decoded to PNG, against the source; with `cvvdp` (a Cvvdp), the pair
    (SSIMULACRA2, CVVDP JOD) from the same PNG"""
    with tempfile.TemporaryDirectory(dir=work) as td:
        png = os.path.join(td, "d.png")
        if not decode_png(a, path, png):
            return None if cvvdp is None else (None, None)
        p = run([a.ssimulacra2, src, png])
        try:
            s = float(p.stdout.strip().split()[-1])
        except (ValueError, IndexError):
            s = None
        return s if cvvdp is None else (s, cvvdp(src, png, path))


# ------------------------------------------------------------------------------------------- CVVDP
_SRGB = []


def srgb8(path, icc=None):
    """an image file as an 8-bit sRGB H x W x 3 array, converted with LittleCMS from its ICC profile (or from `icc`,
    for a decoder that drops the profile); no profile means sRGB"""
    from PIL import Image, ImageCms
    im = Image.open(path)
    icc = im.info.get("icc_profile") or _jpeg_icc(im) or icc
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if icc:
        if not _SRGB:
            _SRGB.append(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")))
        try:  # rendering intent 1: relative colorimetric
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), _SRGB[0],
                                           renderingIntent=1, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError) as e:  # a broken profile, or one for other pixels: use them as they are
            print(f"{path}: ICC profile not applied ({e})", file=sys.stderr, flush=True)
    return np.asarray(im.convert("RGB"))


def _jpeg_icc(im):
    """a JPEG's ICC profile from its APP2 markers.  Pillow only assembles the markers that precede the frame header,
    and some files (4 of the size-range photos) put it after; libjpeg, and so libjxl, libavif and cwebp, read it."""
    chunks = sorted((d[12], d[14:]) for m, d in getattr(im, "applist", []) if m == "APP2" and d[:12] == b"ICC_PROFILE\0")
    return b"".join(c for _, c in chunks) or None


def file_icc(path):
    """the ICC profile inside an encoded WebP (dwebp does not copy it into its PNG); avifdec's PNG keeps its own"""
    if not path.endswith(".webp"):
        return None
    from PIL import Image
    try:
        return Image.open(path).info.get("icc_profile")
    except OSError:
        return None


class Cvvdp:
    """ColorVideoVDP on one display model.  Callable from pool threads: they load and convert the pixels, and the
    metric itself runs one image pair at a time."""

    def __init__(self, display, device=None):
        import torch
        import pycvvdp
        from importlib.metadata import version
        self.metric = pycvvdp.cvvdp(display_name=display, quiet=True, device=torch.device(device) if device else None)
        self.lock = threading.Lock()
        try:  # fail now rather than score every file as None (e.g. a cuDNN that does not match torch's)
            t = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            self.metric.predict(t, t, dim_order="HWC")
        except Exception as e:
            sys.exit(f"cvvdp does not run on {self.metric.device}: {e}")
        m, dp, dg = self.metric, self.metric.display_photometry, self.metric.display_geometry
        dev = m.device
        self.info = dict(
            cvvdp=version("cvvdp"), parameters_version=str(getattr(m, "version", "")), torch=torch.__version__,
            torch_cuda=torch.version.cuda, device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
            display=display, display_name=getattr(dp, "full_name", ""), eotf=getattr(dp, "EOTF", ""),
            peak_cd_m2=getattr(dp, "Y_peak", None), contrast=getattr(dp, "contrast", None),
            ambient_lux=getattr(dp, "E_ambient", None), k_refl=getattr(dp, "k_refl", None),
            resolution=list(getattr(dg, "resolution", ())), distance_m=getattr(dg, "distance_m", None),
            display_size_m=list(getattr(dg, "display_size_m", ()) or ()), pixels_per_degree=float(m.pix_per_deg),
            inputs="8-bit sRGB (ICC profiles converted with LittleCMS, relative colorimetric), 1:1 pixel mapping")

    def __call__(self, src, png, path):
        """JOD of the decoded `png` (of encoded file `path`) against the source image `src`; None on failure"""
        try:
            ref = srgb8(src)
            test = srgb8(png, file_icc(path))
            if test.shape != ref.shape:  # NVENC pads odd sizes to even ones: crop the padding
                h, w = ref.shape[:2]
                if test.shape[0] - h not in (0, 1) or test.shape[1] - w not in (0, 1):
                    raise ValueError(f"decoded {test.shape} vs source {ref.shape}")
                test = test[:h, :w]
            with self.lock:
                jod, _ = self.metric.predict(test, ref, dim_order="HWC")
            return float(jod)
        except Exception as e:  # report, leave the point out of the curve
            print(f"cvvdp failed on {path}: {e}", file=sys.stderr, flush=True)
            return None


def jod_target(quality, sizes, target):
    """for each codec: how many images reach `target` JOD within the tested points, and the bits per pixel (of the
    source's pixels) at which each one does, interpolated along the image's (JOD, log-size) curve as in bd_rate;
    medians over the images that reach it, and over the images that every codec reaches"""
    bpp = {}
    for codec, rows in quality.items():
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append((r["bytes"], r.get("cvvdp")))
        for im, pts in per.items():
            x, y = _curve(pts)
            v = None
            if len(x) and x[-1] >= target:  # at or above it already at the lowest point: that point's size
                lb = y[0] if x[0] >= target else _pchip_eval(x, y, np.array([float(target)]))[0]
                v = float(np.exp(lb)) * 8 / (sizes[im] * 1e6)
            bpp.setdefault(codec, {})[im] = v
    common = set.intersection(*({im for im, v in d.items() if v is not None} for d in bpp.values())) if bpp else set()
    out = {}
    for codec, d in bpp.items():
        hit = [v for v in d.values() if v is not None]
        out[codec] = dict(images=len(d), reached=len(hit), median_bpp=statistics.median(hit) if hit else None,
                          common_images=len(common),
                          common_median_bpp=statistics.median(d[im] for im in common) if common else None,
                          per_image_bpp=d)
    return dict(target=target, codecs=out)


def mean_by_q(quality, target=None):
    """per codec and quality point: images, total bytes, mean SSIMULACRA2, mean JOD and images at or above `target`"""
    out = {}
    for codec, rows in quality.items():
        for q in sorted({r["q"] for r in rows}):
            sel = [r for r in rows if r["q"] == q]
            s2 = [r["ssimulacra2"] for r in sel if r.get("ssimulacra2") is not None]
            jod = [r["cvvdp"] for r in sel if r.get("cvvdp") is not None]
            out.setdefault(codec, {})[str(q)] = dict(
                n=len(sel), bytes=sum(r["bytes"] for r in sel), ssimulacra2=statistics.fmean(s2) if s2 else None,
                cvvdp=statistics.fmean(jod) if jod else None, cvvdp_n=len(jod),
                at_target=sum(j >= target for j in jod) if target is not None else None)
    return out


def encode_cpu_set(a, codec, files, qualities, outdir, tune_iq):
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    for src in files:
        base = os.path.basename(src)
        for q in qualities:
            if codec.startswith(("aom_s", "svt_p")):
                dst = os.path.join(outdir, f"{base}.q{q}.avif")
                jobs.append((src, q, dst, avifenc_cmd(a, src, dst, q, int(codec[5:]), tune_iq, codec[:3])))
            else:
                dst = os.path.join(outdir, f"{base}.q{q}.webp")
                jobs.append((src, q, dst, [a.cwebp, "-quiet", "-q", str(q), "-m", "4", "-metadata", "icc", src, "-o", dst]))
    with cf.ThreadPoolExecutor(a.jobs if codec == "webp" else avifenc_parallel(a, a.jobs)) as ex:
        res = list(ex.map(lambda j: (j, run(j[3]).returncode), jobs))
    return [(src, q, dst) for (src, q, dst, _), rc in res if rc == 0 and os.path.exists(dst)]


def encode_gpu_set(a, files, qualities, outdir):
    os.makedirs(outdir, exist_ok=True)
    lst = os.path.join(outdir, "inputs.txt")
    open(lst, "w").write("\n".join(files) + "\n")
    qi = {q: qindex_for_quality(q) for q in qualities}
    p = run([a.av1gpu, "--batch", lst, "--outdir", outdir, "--qindex", ",".join(str(v) for v in qi.values()),
             "--name-q"] + gpu_format(a))
    if p.returncode:
        sys.exit(f"av1gpu failed:\n{p.stderr[-3000:]}")
    out = []
    for src in files:
        for q, v in qi.items():
            dst = os.path.join(outdir, f"{os.path.basename(src)}.q{v}.avif")
            if os.path.exists(dst):
                out.append((src, q, dst))
    return out


# ------------------------------------------------------------------------------------------- speed
SIZE_CLASSES = [("< 1 MP", 0, 1), ("1-5 MP", 1, 5), ("5-10 MP", 5, 10), (">= 10 MP", 10, float("inf"))]


def speed_set(files, sizes, min_mp, work):
    """the image list repeated (as uniquely named symlinks) until it holds at least min_mp megapixels: a short
    run is dominated by CUDA start-up and by filling the encoder pipeline (the speed table in docs/RESULTS.md
    comes from 1,240 MP runs)"""
    d = os.path.join(work, "speedset")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    out, mp, rep = [], 0.0, 0
    while not out or mp < min_mp:
        for f in files:
            link = os.path.join(d, f"r{rep}_{os.path.basename(f)}")
            os.symlink(os.path.abspath(f), link)
            out.append(link)
            mp += sizes[os.path.basename(f)]
        rep += 1
    return d, out, mp, rep


def gpu_speed(a, image_dir, work):
    runs = []
    for i in range(a.repeats + 1):
        out = os.path.join(work, "speed_gpu")
        shutil.rmtree(out, ignore_errors=True)
        p = run([a.av1gpu, "--batch", image_dir, "--outdir", out, "--stats"] + gpu_format(a))
        if p.returncode:
            sys.exit(f"av1gpu failed:\n{p.stderr[-3000:]}")
        st = json.loads(p.stdout.strip().splitlines()[-1])
        if i:  # run 0 is the warm-up (driver JIT caches, file cache)
            runs.append(st)
    med = lambda k: statistics.median(r[k] for r in runs)
    return dict(codec=f"av1gpu ({a.gpu_label}{format_label(a)})", images=runs[0]["images"], mp=runs[0]["mp"], mp_s=med("mp_s"),
                img_s=med("img_s"), cpu_cores=med("cpu_cores"), wall_s=med("wall_ms") / 1e3, bytes=runs[0]["bytes"],
                runs=runs)


def cpu_speed(a, files, speed, tune_iq, mp, work, codec="aom"):
    out = os.path.join(work, f"speed_{codec}_{speed}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    jobs = [avifenc_cmd(a, f, os.path.join(out, os.path.basename(f) + ".avif"), 60, speed, tune_iq, codec)
            for f in files]
    par, threads = avifenc_parallel(a, os.cpu_count()), avifenc_threads(a)
    t0 = time.time()
    with cf.ThreadPoolExecutor(par) as ex:
        rcs = list(ex.map(lambda c: run(c).returncode, jobs))
    wall = time.time() - t0
    nbytes = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "*.avif")))
    name = f"SVT-AV1 preset {speed}" if codec == "svt" else f"libaom speed {speed}"
    tiles = ", no tiling" if getattr(a, "no_tiling", False) else ""
    return dict(codec=f"{name}, {par} parallel avifenc -j {threads}{tiles}{format_label(a)}", images=rcs.count(0), mp=mp,
                mp_s=mp / wall, img_s=rcs.count(0) / wall, cpu_cores=float(min(os.cpu_count(), par * threads)),
                wall_s=wall, bytes=nbytes)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(HERE, "images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--av1gpu", default=os.path.join(ROOT, "build", "av1gpu"))
    ap.add_argument("--gpu-label", default="default settings", help="av1gpu settings, for the summary")
    ap.add_argument("--avifenc", default="avifenc")
    ap.add_argument("--avifdec", default="avifdec")
    ap.add_argument("--cwebp", default="cwebp")
    ap.add_argument("--dwebp", default="dwebp")
    ap.add_argument("--ssimulacra2", default="ssimulacra2")
    ap.add_argument("--quality", default="30,40,50,60,70,80", help="avifenc -q / av1gpu --quality points")
    ap.add_argument("--webp-quality", default="50,60,70,80,90,95")
    ap.add_argument("--aom-speeds", default="6,9", help="libaom speeds to compare (6 is the BD-rate anchor and always runs)")
    ap.add_argument("--svt-presets", default="", help="SVT-AV1 presets to compare (needs avifenc built with SVT-AV1)")
    ap.add_argument("--svt-tune", default="", help="SVT-AV1 tune= value passed through avifenc -a (default: SVT's own)")
    ap.add_argument("--avifenc-threads", type=int, default=1,
                    help="avifenc -j N for libaom/SVT-AV1, with cpu_count // N encodes at once (default 1)")
    ap.add_argument("--no-tiling", action="store_true",
                    help="one tile per image (avifenc --tilerowslog2 0 --tilecolslog2 0) instead of avifenc's automatic tiling")
    ap.add_argument("--yuv", choices=("420", "444"), default="420", help="chroma format of every AV1 encode (default 420)")
    ap.add_argument("--depth", type=int, choices=(8, 10), default=8, help="bit depth of every AV1 encode (default 8)")
    ap.add_argument("--min-mp", type=float, default=1000, help="speed runs repeat the set up to this many megapixels")
    ap.add_argument("--repeats", type=int, default=3, help="timed av1gpu runs (after one warm-up)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the encoded files under OUT/files")
    ap.add_argument("--avifdec-codec", default="", help="avifdec -c CODEC for decoding AVIF before scoring (e.g. dav1d)")
    ap.add_argument("--cvvdp", default="", metavar="DISPLAY",
                    help="also score with ColorVideoVDP on this display model (e.g. standard_fhd; needs pip install cvvdp)")
    ap.add_argument("--cvvdp-device", default="", help="torch device for CVVDP (default: cuda if available, else cpu)")
    ap.add_argument("--cvvdp-target", type=float, default=9.5, help="JOD for the bits-per-pixel-to-reach table")
    ap.add_argument("--cvvdp-min-overlap", type=float, default=0.05,
                    help="least JOD range two curves must share for a BD-rate; the default matches SSIMULACRA2's 2 "
                         "points (libaom speed 6 spans about 35 SSIMULACRA2 points per JOD from q30 to q80)")
    a = ap.parse_args()
    a.aom_speeds = sorted({6} | {int(x) for x in a.aom_speeds.split(",")})
    a.svt_presets = [int(x) for x in a.svt_presets.split(",") if x]
    if a.avifenc_threads < 1:
        sys.exit("--avifenc-threads must be at least 1")
    cv = Cvvdp(a.cvvdp, a.cvvdp_device or None) if a.cvvdp and not a.skip_quality else None

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"no JPEGs in {a.images} (run bench/fetch_images.py first)")
    for t in ("avifenc", "avifdec", "cwebp", "dwebp", "ssimulacra2") + (() if a.skip_gpu else ("av1gpu",)):
        if not shutil.which(getattr(a, t)):
            sys.exit(f"{t} not found ({getattr(a, t)}); install it or pass --{t}")
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "files")
    os.makedirs(work, exist_ok=True)
    from PIL import Image
    sizes = {os.path.basename(f): float(np.prod(Image.open(f).size)) / 1e6 for f in files}
    mp = sum(sizes.values())
    tune_iq = avifenc_supports_tune_iq(a, files[0], work)
    res = dict(environment=environment(a), images=len(files), megapixels=mp, avifenc_tune_iq=tune_iq,
               quality_points=a.quality, speed=[], quality={}, bd_rate={})
    if a.avifenc_threads != 1 or a.no_tiling:
        res.update(avifenc_threads=a.avifenc_threads, avifenc_parallel=avifenc_parallel(a, os.cpu_count()),
                   no_tiling=a.no_tiling)
    if (a.yuv, a.depth) != ("420", 8):
        res.update(yuv=a.yuv, depth=a.depth)
    if a.avifdec_codec:
        res["avifdec_codec"] = a.avifdec_codec
    print(f"{len(files)} images, {mp:.1f} MP; avifenc tune=iq: {'yes' if tune_iq else 'no (libaom too old)'}", flush=True)

    if not a.skip_speed:
        sdir, sfiles, smp, reps = speed_set(files, sizes, a.min_mp, work)
        res["speed_workload"] = dict(images=len(sfiles), megapixels=smp, repeats=reps)
        print(f"speed workload: the set {reps}x, {len(sfiles)} images, {smp:.0f} MP", flush=True)
        if not a.skip_gpu:
            res["speed"].append(gpu_speed(a, sdir, work))
            print("speed:", res["speed"][-1]["codec"], f"{res['speed'][-1]['mp_s']:.1f} MP/s", flush=True)
        for s in a.aom_speeds:
            res["speed"].append(cpu_speed(a, sfiles, s, tune_iq, smp, work))
            print("speed:", res["speed"][-1]["codec"], f"{res['speed'][-1]['mp_s']:.1f} MP/s", flush=True)
        for p in a.svt_presets:
            res["speed"].append(cpu_speed(a, sfiles, p, tune_iq, smp, work, "svt"))
            print("speed:", res["speed"][-1]["codec"], f"{res['speed'][-1]['mp_s']:.1f} MP/s", flush=True)

    if not a.skip_quality:
        qs = [int(x) for x in a.quality.split(",")]
        wq = [int(x) for x in a.webp_quality.split(",")]
        enc = {}
        if not a.skip_gpu:
            enc["av1gpu"] = encode_gpu_set(a, files, qs, os.path.join(work, "av1gpu"))
        for codec in [f"aom_s{s}" for s in a.aom_speeds] + [f"svt_p{p}" for p in a.svt_presets]:
            enc[codec] = encode_cpu_set(a, codec, files, qs, os.path.join(work, codec), tune_iq)
        enc["webp"] = encode_cpu_set(a, "webp", files, wq, os.path.join(work, "webp"), False)
        for codec, outs in enc.items():
            print(f"scoring {codec}: {len(outs)} files", flush=True)
            with cf.ThreadPoolExecutor(a.jobs) as ex:
                scores = list(ex.map(lambda t: score(a, t[0], t[2], work, cv), outs))
            if cv:
                res["quality"][codec] = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst),
                                              ssimulacra2=s, cvvdp=j) for (src, q, dst), (s, j) in zip(outs, scores)]
            else:
                res["quality"][codec] = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst), ssimulacra2=s)
                                         for (src, q, dst), s in zip(outs, scores)]
        res["bd_rate"], by_size = bd_table(res["quality"], sizes)
        if by_size:
            res["bd_rate_by_size"] = by_size
        if cv:
            res["cvvdp"] = dict(cv.info, target=a.cvvdp_target, min_overlap=a.cvvdp_min_overlap)
            res["bd_rate_cvvdp"], res["bd_rate_cvvdp_by_size"] = bd_table(res["quality"], sizes, "cvvdp",
                                                                          a.cvvdp_min_overlap)
            res["cvvdp_target"] = jod_target(res["quality"], sizes, a.cvvdp_target)
            res["by_quality"] = mean_by_q(res["quality"], a.cvvdp_target)

    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    write_summary(res, os.path.join(a.out, "summary.md"))
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(open(os.path.join(a.out, "summary.md")).read())


def write_summary(res, path):
    e = res["environment"]
    L = [f"# av1gpu benchmark", "",
         f"- GPU: {e['gpu']}", f"- CPU: {e['cpu']} ({e['cpus']} logical CPUs)",
         f"- {e['avifenc']}; tune=iq: {'yes' if res['avifenc_tune_iq'] else 'no'}", f"- cwebp {e['cwebp']}",
         f"- {res['images']} images, {res['megapixels']:.1f} MP"]
    if res.get("yuv"):
        L.append(f"- AV1 encodes {res['depth']}-bit {res['yuv']}" +
                 (f"; decoded with avifdec -c {res['avifdec_codec']}" if res.get("avifdec_codec") else ""))
    if res.get("speed_workload"):
        w = res["speed_workload"]
        L.append(f"- speed runs: the set repeated {w['repeats']}x ({w['images']} images, {w['megapixels']:.0f} MP)")
    L.append("")
    if res["speed"]:
        L += ["## Throughput (quality 60)", "", "| Encoder | MP/s | images/s | CPU cores used | output MB |",
              "|---|---:|---:|---:|---:|"]
        for s in res["speed"]:
            L.append(f"| {s['codec']} | {s['mp_s']:.1f} | {s['img_s']:.1f} | {s['cpu_cores']:.1f} | {s['bytes'] / 1e6:.2f} |")
        L.append("")
    L += summary_bd(res["bd_rate"], res.get("bd_rate_by_size"), "SSIMULACRA2")
    if res.get("cvvdp"):
        L += summary_cvvdp(res)
    open(path, "w").write("\n".join(L))


def summary_bd(bd, by_size, metric, suffix=""):
    L = []
    if bd:
        L += [f"## Compression (BD-rate on {metric}, per image; negative = smaller files at equal quality){suffix}", "",
              "| Codec | vs | images | median | mean |", "|---|---|---:|---:|---:|"]
        for k, v in bd.items():
            c, anchor = k.split("_vs_")
            L.append(f"| {c} | {anchor} | {v['n']} | {100 * v['median']:+.1f}% | {100 * v['mean']:+.1f}% |")
        L.append("")
    if by_size:
        L += [f"## BD-rate{'' if metric == 'SSIMULACRA2' else ' on ' + metric} vs libaom s6 by image size "
              "(median; images in brackets)", "",
              "| Codec | " + " | ".join(c for c, _, _ in SIZE_CLASSES) + " |", "|---|" + "---:|" * len(SIZE_CLASSES)]
        for k, by in by_size.items():
            cells = [f"{100 * by[c]['median']:+.1f}% ({by[c]['n']})" if c in by else "—" for c, _, _ in SIZE_CLASSES]
            L.append(f"| {k.split('_vs_')[0]} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def summary_cvvdp(res):
    c = res["cvvdp"]
    L = [f"## CVVDP", "",
         f"- cvvdp {c['cvvdp']} (parameters {c['parameters_version']}), torch {c['torch']} (CUDA {c['torch_cuda']}) "
         f"on {c['device']}",
         f"- display `{c['display']}`: {c['display_name']}; {c['resolution'][0]} x {c['resolution'][1]}, "
         f"{c['peak_cd_m2']} cd/m^2, contrast {c['contrast']}:1, {c['ambient_lux']} lux, {c['distance_m']} m, "
         f"{c['pixels_per_degree']:.1f} pixels per degree; EOTF {c['eotf']}",
         f"- inputs: {c['inputs']}", ""]
    L += summary_bd(res["bd_rate_cvvdp"], res.get("bd_rate_cvvdp_by_size"), "CVVDP JOD",
                    f" (curves must share {c['min_overlap']} JOD)")
    t = res["cvvdp_target"]
    any_codec = next(iter(t["codecs"].values()), None)
    if any_codec:
        L += [f"## Bits per pixel to reach {t['target']} JOD", "",
              f"Interpolated per image within the tested qualities. The last column is over the "
              f"{any_codec['common_images']} images that every codec reaches.", "",
              "| Codec | images reaching | median bpp | median bpp, common images |", "|---|---:|---:|---:|"]
        f = lambda v: f"{v:.3f}" if v is not None else "—"
        for codec, v in t["codecs"].items():
            L.append(f"| {codec} | {v['reached']}/{v['images']} | {f(v['median_bpp'])} | {f(v['common_median_bpp'])} |")
        L.append("")
    L += ["## Mean scores per quality point", "",
          f"| Codec | q | images | MB | mean SSIMULACRA2 | mean JOD | images >= {t['target']} JOD |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    f = lambda v, d: f"{v:.{d}f}" if v is not None else "—"
    for codec, by in res["by_quality"].items():
        for q, v in by.items():
            L.append(f"| {codec} | {q} | {v['n']} | {v['bytes'] / 1e6:.2f} | {f(v['ssimulacra2'], 2)} | "
                     f"{f(v['cvvdp'], 3)} | {v['at_target']} |")
    L.append("")
    return L


if __name__ == "__main__":
    main()
