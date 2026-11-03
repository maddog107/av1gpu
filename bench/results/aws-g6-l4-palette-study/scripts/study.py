#!/usr/bin/env python3
"""Palette / IntraBC study: encode the sets with libaom (avifenc) in the screen-content configurations, with av1gpu
and (lossless) cwebp, then score, verify and inspect every file.  Run on the instance, after prep_sets.py.

  study.py encode-gpu | encode-aom | encode-ll | score | all      (outputs under ~/pal/out, rows in ~/pal/rows/*.jsonl)

libaom 3.13.1 through avifenc 1.4.0, the way bench/bench.py runs it (-s 6 -j 1 -a tune=iq --ignore-exif --ignore-xmp,
all-intra), one avifenc per vCPU.  Only the screen-content switches differ between configurations:
  A   palette and IntraBC off (-a enable-palette=0 -a enable-intrabc=0); libaom's screen-content detection still runs
  B   libaom's default: its own detection (tune=iq selects the anti-aliasing-aware mode 2) and trial encode decide
  C   screen-content tools forced on (-a tune-content=screen), IntraBC off (-a enable-intrabc=0): palette only
  D   screen-content tools forced on, IntraBC allowed (-a tune-content=screen)
  C0  C with palette off too: what forcing the screen-content mode changes besides palette (its speed features)
Lossy: 8-bit 4:2:0 (A B C D C0) and 10-bit 4:4:4 (A C D), quality 30 45 60 75 90.  av1gpu: the default and
--yuv 444 --depth 10 at the same quality points (bench.py's quality -> qindex table).  Lossless (PNG sets): avifenc -l
in A B C D, A and D with --cicp 1/13/16 (YCgCo-Re), av1gpu --lossless and --lossless --ycgco-re, cwebp -lossless
(default and -z 9).  Scores: SSIMULACRA2 and butteraugli (libjxl 0.11.1, 3-norm) of the dav1d decode against the
source; lossless files are decoded and compared with the source pixels.  pstat (libaom inspection) gives every libaom
file's frame flags and palette / IntraBC block counts."""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

H = os.path.expanduser("~")
PAL = f"{H}/pal"
SETS = f"{PAL}/sets"
OUT = os.environ.get("STUDY_OUT", f"{PAL}/out")      # STUDY_OUT / STUDY_ROWS / STUDY_AV1GPU: another av1gpu build
ROWS = os.environ.get("STUDY_ROWS", f"{PAL}/rows")   # (the palette branch) into its own outputs and rows
AVIFENC, AVIFDEC = f"{H}/opt/bin/avifenc", f"{H}/opt/bin/avifdec"
SSIM2, BUTTER, PSTAT = f"{H}/opt/bin/ssimulacra2", f"{H}/opt/bin/butteraugli_main", f"{H}/opt/bin/pstat"
AV1GPU = os.environ.get("STUDY_AV1GPU", f"{H}/av1gpu/build/av1gpu")
NPROC = os.cpu_count()
QS = [30, 45, 60, 75, 90]
LOSSY_SETS = ["comics", "flat_png", "flat_jpg", "painted", "screen", "photo"]
LL_SETS = ["flat_png", "screen", "photo"]
CFG = {
    "A": ["-a", "enable-palette=0", "-a", "enable-intrabc=0"],
    "B": [],
    "C": ["-a", "tune-content=screen", "-a", "enable-intrabc=0"],
    "D": ["-a", "tune-content=screen"],
    "C0": ["-a", "tune-content=screen", "-a", "enable-palette=0", "-a", "enable-intrabc=0"],
}
FMT = {"420": ["-y", "420"], "444_10": ["-y", "444", "-d", "10"]}
LOSSY_PLAN = [("420", c) for c in ("A", "B", "C", "D", "C0")] + [("444_10", c) for c in ("A", "C", "D")]
LL_PLAN = [("gbr", c) for c in ("A", "B", "C", "D")] + [("ycgco", c) for c in ("A", "D")]

sys.path.insert(0, f"{H}/av1gpu/bench")
from bench import qindex_for_quality  # noqa: E402


def images(s, png_only=False):
    d = os.path.join(SETS, s)
    fs = sorted(f for f in os.listdir(d) if not png_only or f.endswith(".png"))
    return [os.path.join(d, f) for f in fs]


def mp(path):
    im = Image.open(path)
    return im.width * im.height / 1e6


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pool(jobs, fn, workers=NPROC, label=""):
    """run fn(job) over jobs (largest first), report progress; returns the results"""
    jobs = sorted(jobs, key=lambda j: -j.get("mp", 0))
    out, t0, n = [], time.time(), len(jobs)
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, jobs)):
            out.append(r)
            if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
                print(f"{label} {i + 1}/{n} {time.time() - t0:.0f} s", flush=True)
    return out


def write_rows(name, rows):
    os.makedirs(ROWS, exist_ok=True)
    with open(f"{ROWS}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------------------------------------ encoders
def aom_lossy_job(j):
    if not os.path.exists(j["dst"]):
        os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
        cmd = [AVIFENC, "-q", str(j["q"])] + FMT[j["fmt"]] + ["-j", "1", "--ignore-exif", "--ignore-xmp", "-s", "6",
                                                              "-a", "tune=iq"] + CFG[j["cfg"]] + [j["src"], j["dst"]]
        t = time.time()
        p = run(cmd)
        j["enc_s"] = time.time() - t
        if p.returncode:
            j["error"] = p.stderr[-500:]
    return j


def encode_aom():
    jobs = []
    for s in LOSSY_SETS:
        for src in images(s):
            m = mp(src)
            for fmt, c in LOSSY_PLAN:
                for q in QS:
                    jobs.append(dict(kind="lossy", enc="aom", fmt=fmt, cfg=c, set=s, image=os.path.basename(src), q=q, mp=m,
                                     src=src, dst=f"{OUT}/aom/{fmt}_{c}/{s}/{os.path.basename(src)}.q{q}.avif"))
    rows = pool(jobs, aom_lossy_job, label="aom lossy")
    write_rows("enc_aom", rows)


def aom_ll_job(j):
    if not os.path.exists(j["dst"]):
        os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
        cmd = [AVIFENC, "-l", "-j", "1", "-s", "6"] + CFG[j["cfg"]] + (["--cicp", "1/13/16"] if j["fmt"] == "ycgco" else []) + \
            [j["src"], j["dst"]]
        t = time.time()
        p = run(cmd)
        j["enc_s"] = time.time() - t
        if p.returncode:
            j["error"] = p.stderr[-500:]
    return j


def webp_ll_job(j):
    if not os.path.exists(j["dst"]):
        os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
        t = time.time()
        p = run(["cwebp", "-quiet", "-lossless"] + (["-z", "9"] if j["cfg"] == "z9" else []) + [j["src"], "-o", j["dst"]])
        j["enc_s"] = time.time() - t
        if p.returncode:
            j["error"] = p.stderr[-500:]
    return j


def encode_ll():
    jobs = []
    for s in LL_SETS:
        for src in images(s, png_only=True):
            m, b = mp(src), os.path.basename(src)
            for fmt, c in LL_PLAN:
                jobs.append(dict(kind="lossless", enc="aom", fmt=fmt, cfg=c, set=s, image=b, mp=m, src=src,
                                 dst=f"{OUT}/aom_ll/{fmt}_{c}/{s}/{b}.avif"))
            for c in ("default", "z9"):
                jobs.append(dict(kind="lossless", enc="webp", fmt="argb", cfg=c, set=s, image=b, mp=m, src=src,
                                 dst=f"{OUT}/webp_ll/{c}/{s}/{b}.webp"))
    aom = pool([j for j in jobs if j["enc"] == "aom"], aom_ll_job, label="aom lossless")
    webp = pool([j for j in jobs if j["enc"] == "webp"], webp_ll_job, label="cwebp lossless")
    # av1gpu lossless, both colour spaces
    gpu = []
    for mode, extra in (("gbr", []), ("ycgco", ["--ycgco-re"])):
        for s in LL_SETS:
            srcs = images(s, png_only=True)
            od = f"{OUT}/gpu_ll/{mode}/{s}"
            os.makedirs(od, exist_ok=True)
            lst = f"{od}/inputs.txt"
            open(lst, "w").write("\n".join(srcs) + "\n")
            t = time.time()
            p = run([AV1GPU, "--batch", lst, "--outdir", od, "--lossless"] + extra, env=dict(os.environ, GP_ENCODE_MP="12"))
            wall = time.time() - t
            print(f"av1gpu lossless {mode} {s}: exit {p.returncode}, {wall:.1f} s", flush=True)
            for src in srcs:
                gpu.append(dict(kind="lossless", enc="gpu", fmt=mode, cfg="lossless", set=s, image=os.path.basename(src),
                                mp=mp(src), src=src, dst=f"{od}/{os.path.basename(src)}.avif", batch_wall_s=wall,
                                **({"error": p.stderr[-500:]} if p.returncode else {})))
    write_rows("enc_ll", aom + webp + gpu)


def encode_gpu():
    rows = []
    qi = {q: qindex_for_quality(q) for q in QS}
    for fmt, extra in (("420", []), ("444_10", ["--yuv", "444", "--depth", "10"])):
        for s in LOSSY_SETS:
            srcs = images(s)
            od = f"{OUT}/gpu/{fmt}/{s}"
            os.makedirs(od, exist_ok=True)
            lst = f"{od}/inputs.txt"
            open(lst, "w").write("\n".join(srcs) + "\n")
            t = time.time()
            p = run([AV1GPU, "--batch", lst, "--outdir", od, "--qindex", ",".join(str(v) for v in qi.values()), "--name-q"]
                    + extra + os.environ.get("STUDY_GPU_EXTRA", "").split())   # e.g. "--palette on"
            wall = time.time() - t
            print(f"av1gpu {fmt} {s}: exit {p.returncode}, {wall:.1f} s", flush=True)
            for src in srcs:
                for q, v in qi.items():
                    rows.append(dict(kind="lossy", enc="gpu", fmt=fmt, cfg="default", set=s, image=os.path.basename(src), q=q,
                                     qindex=v, mp=mp(src), src=src, dst=f"{od}/{os.path.basename(src)}.q{v}.avif",
                                     batch_wall_s=wall, **({"error": p.stderr[-500:]} if p.returncode else {})))
    write_rows("enc_gpu", rows)


# ------------------------------------------------------------------------------------------------ scoring
def pstat(path):
    with tempfile.TemporaryDirectory() as td:
        ivf = os.path.join(td, "a.ivf")
        p = run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "avif2ivf.py"), path, ivf])
        if p.returncode:
            return None
        p = run([PSTAT, ivf])
        try:
            return json.loads(p.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None


def score_lossy(j):
    if j.get("error") or not os.path.exists(j["dst"]):
        j["missing"] = True
        return j
    j["bytes"] = os.path.getsize(j["dst"])
    with tempfile.TemporaryDirectory(dir=f"{PAL}/tmp") as td:
        png = os.path.join(td, "d.png")
        p = run([AVIFDEC, "-c", "dav1d", "--png-compress", "1"] + (["-d", "16"] if j["fmt"] == "444_10" else []) +
                [j["dst"], png])
        if p.returncode:
            j["decode_error"] = p.stderr[-300:]
            return j
        p = run([SSIM2, j["src"], png])
        try:
            j["ssimulacra2"] = float(p.stdout.strip().split()[-1])
        except (ValueError, IndexError):
            j["ssimulacra2"] = None
        if j.get("butter"):
            p = run([BUTTER, j["src"], png])
            for line in p.stdout.splitlines():
                if line.startswith("3-norm"):
                    j["butteraugli"] = float(line.split()[-1])
                elif line.strip() and "butteraugli_max" not in j:
                    try:
                        j["butteraugli_max"] = float(line.strip())
                    except ValueError:
                        pass
    if j["enc"] == "aom":
        j["pstat"] = pstat(j["dst"])
    return j


def pixels(path, depth8=True):
    im = Image.open(path)
    return np.asarray(im.convert("RGB"))


def score_ll(j):
    if j.get("error") or not os.path.exists(j["dst"]):
        j["missing"] = True
        return j
    j["bytes"] = os.path.getsize(j["dst"])
    with tempfile.TemporaryDirectory(dir=f"{PAL}/tmp") as td:
        png = os.path.join(td, "d.png")
        if j["dst"].endswith(".webp"):
            p = run(["dwebp", j["dst"], "-o", png])
        else:
            p = run([AVIFDEC, "-c", "dav1d", "-d", "8", "--png-compress", "1", j["dst"], png])
        if p.returncode:
            j["decode_error"] = p.stderr[-300:]
            return j
        j["exact"] = bool(np.array_equal(pixels(png), pixels(j["src"])))
    if j["enc"] == "aom":
        j["pstat"] = pstat(j["dst"])
    return j


def load(name):
    p = f"{ROWS}/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def score():
    os.makedirs(f"{PAL}/tmp", exist_ok=True)
    lossy = load("enc_gpu") + load("enc_aom")
    lossy = [j for j in lossy if j["set"] in os.environ.get("STUDY_SCORE_SETS", " ".join(LOSSY_SETS)).split()]
    for j in lossy:   # butteraugli for the 8-bit 4:2:0 A / C / D files and av1gpu's default
        j["butter"] = (j["fmt"] == "420" and (j["enc"] == "gpu" or j["cfg"] in ("A", "C", "D")))
    write_rows("scored_lossy", pool(lossy, score_lossy, label="score lossy"))
    write_rows("scored_ll", pool(load("enc_ll"), score_ll, label="score lossless"))


if __name__ == "__main__":
    step = sys.argv[1]
    t0 = time.time()
    if os.environ.get("STUDY_SETS"):
        LOSSY_SETS[:] = os.environ["STUDY_SETS"].split()
        LL_SETS[:] = [x for x in LL_SETS if x in LOSSY_SETS]
    if step in ("encode-gpu", "all"):
        encode_gpu()
    if step in ("encode-aom", "all"):
        encode_aom()
    if step in ("encode-ll", "all"):
        encode_ll()
    if step in ("score", "all"):
        score()
    print(f"{step} done in {time.time() - t0:.0f} s", flush=True)
