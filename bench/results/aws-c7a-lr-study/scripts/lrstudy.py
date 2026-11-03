#!/usr/bin/env python3
"""Loop-restoration study: encode the photo sets with libaom 3.13.1 (avifenc 1.4.0) with and without loop restoration,
then score and inspect every file.  Run on the instance after setup.sh and build_lr.sh.

  lrstudy.py encode [CFG ...]      encodes -> ~/lr/out/<fmt>/<cfg>/<set>/<image>.q<Q>.avif, rows ~/lr/rows/enc.jsonl
  lrstudy.py score                 dav1d decode (avifdec -c dav1d) -> SSIMULACRA2 (libjxl 0.11.1) and RGB PSNR
                                   against the source, plus lrstat (libaom inspection) -> ~/lr/rows/score.jsonl

libaom runs the way bench/bench.py runs it: avifenc -q Q -y 420 -j 1 -s S -a tune=iq --ignore-exif --ignore-xmp,
all-intra, one avifenc per vCPU (all configurations' jobs in one pool, largest first, so they share the machine
alike).  Configurations (stock = ~/opt/bin/avifenc, patched = ~/opt-lr/bin/avifenc with patch_lrstudy.py):
  s6        stock, speed 6, -a enable-restoration=0              the reference (av1gpu is +0.1 % against it)
  s6r1      stock, speed 6, -a enable-restoration=1              libaom's default; checked byte-identical to s6
  p6keep    patched speed 6, LRSTUDY=keep                        LR as libaom searches it at speed 4, at speed 6
  p6full    patched speed 6, LRSTUDY=full                        LR with every pruning off and the unit-size search
  p6wiener  patched speed 6, LRSTUDY=wiener                      full, Wiener only
  p6sgr     patched speed 6, LRSTUDY=sgr                         full, self-guided only
  s4 / s4r1 stock, speed 4, enable-restoration=0 / 1             libaom's fastest all-intra speed with LR on
  p4full    patched speed 4, LRSTUDY=full
Quality 30, 45, 60, 75, 90 (q60 = qindex 120, av1gpu's default).  Formats: 420 (all sets), 444 (-y 444; Kodak and
the standard set).  CPU time per encode is the child's rusage (user + system), as a proxy for LR's cost."""
import concurrent.futures as cf
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time

H = os.path.expanduser("~")
LR = f"{H}/lr"
OUT, ROWS = f"{LR}/out", f"{LR}/rows"
E, P = f"{H}/opt/bin/avifenc", f"{H}/opt-lr/bin/avifenc"
AVIFDEC, SSIM2, LRSTAT = f"{H}/opt/bin/avifdec", f"{H}/opt/bin/ssimulacra2", f"{H}/opt/bin/lrstat"
QS = [30, 45, 60, 75, 90]
CFG = {
    "s6": (E, 6, 0, ""), "s6r1": (E, 6, 1, ""),
    "p6keep": (P, 6, 1, "keep"), "p6full": (P, 6, 1, "full"), "p6wiener": (P, 6, 1, "wiener"), "p6sgr": (P, 6, 1, "sgr"),
    "s4": (E, 4, 0, ""), "s4r1": (E, 4, 1, ""), "p4full": (P, 4, 1, "full"),
}
SETS = {"std": f"{H}/img/std", "gamut": f"{H}/img/gamut", "kodak": f"{H}/img/kodak"}
PLAN = {"420": (list(CFG), ["std", "gamut", "kodak"]),
        "444": (["s6", "p6keep", "p6full", "s4", "s4r1"], ["std", "kodak"])}


def images(s):
    return sorted(os.path.join(SETS[s], f) for f in os.listdir(SETS[s]) if f.endswith((".jpg", ".png")))


def mp(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size[0] * im.size[1] / 1e6


def run_timed(cmd, env):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    _, status, ru = os.wait4(p.pid, 0)
    err = p.stderr.read().decode(errors="replace")
    p.stderr.close()
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, time.perf_counter() - t0, ru.ru_utime + ru.ru_stime, err


def encode(cfgs):
    os.makedirs(ROWS, exist_ok=True)
    jobs = []
    for fmt, (fcfgs, fsets) in PLAN.items():
        for c in fcfgs:
            if cfgs and c not in cfgs:
                continue
            for s in fsets:
                for src in images(s):
                    for q in QS:
                        dst = f"{OUT}/{fmt}/{c}/{s}/{os.path.basename(src)}.q{q}.avif"
                        if not os.path.exists(dst):
                            jobs.append((fmt, c, s, src, q, dst))
    sizes = {}
    for j in jobs:
        if j[3] not in sizes:
            sizes[j[3]] = mp(j[3])
    random.seed(1)
    random.shuffle(jobs)
    jobs.sort(key=lambda j: -sizes[j[3]] * (4 if j[1].startswith(("s4", "p4")) else 1))
    print(f"{len(jobs)} encodes", flush=True)

    def one(j):
        fmt, c, s, src, q, dst = j
        exe, speed, rst, mode = CFG[c]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cmd = [exe, "-q", str(q), "-y", fmt, "-j", "1", "-s", str(speed), "-a", "tune=iq", "-a",
               f"enable-restoration={rst}", "--ignore-exif", "--ignore-xmp", src, dst + ".part"]
        env = dict(os.environ, LRSTUDY=mode)
        rc, wall, cpu, err = run_timed(cmd, env)
        if rc == 0:
            os.replace(dst + ".part", dst)
        return dict(fmt=fmt, cfg=c, set=s, image=os.path.basename(src), q=q, mp=sizes[src], rc=rc, wall=wall, cpu=cpu,
                    bytes=os.path.getsize(dst) if rc == 0 else None, err=err[-300:] if rc else "")

    t0 = time.time()
    with open(f"{ROWS}/enc.jsonl", "a") as f, cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            f.write(json.dumps(r) + "\n")
            if r["rc"]:
                print("FAIL", r, flush=True)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"encode done {time.time() - t0:.0f}s", flush=True)


def psnr_rgb(src, png):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def score():
    done = set()
    sp = f"{ROWS}/score.jsonl"
    if os.path.exists(sp):
        for l in open(sp):
            r = json.loads(l)
            done.add(r["path"])
    jobs = []
    for fmt, (fcfgs, fsets) in PLAN.items():
        for c in fcfgs:
            for s in fsets:
                for src in images(s):
                    for q in QS:
                        p = f"{OUT}/{fmt}/{c}/{s}/{os.path.basename(src)}.q{q}.avif"
                        if os.path.exists(p) and p not in done:
                            jobs.append((fmt, c, s, src, q, p))
    jobs.sort(key=lambda j: -os.path.getsize(j[3]))
    print(f"{len(jobs)} to score", flush=True)
    sys.path.insert(0, f"{H}/bench")
    import struct
    import avif2ivf  # the palette study's AVIF -> IVF (primary item's OBUs)  # noqa: E402

    def avif_to_ivf(p, ivf):
        b = open(p, "rb").read()
        obus = avif2ivf.primary_item(b)
        w, h = avif2ivf.ispe(b)
        hdr = b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", w & 0xFFFF, h & 0xFFFF, 30, 1, 1, 0)
        open(ivf, "wb").write(hdr + struct.pack("<IQ", len(obus), 0) + obus)

    def one(j):
        fmt, c, s, src, q, p = j
        r = dict(fmt=fmt, cfg=c, set=s, image=os.path.basename(src), q=q, path=p, bytes=os.path.getsize(p))
        with tempfile.TemporaryDirectory(dir=f"{LR}/tmp") as td:
            png = os.path.join(td, "d.png")
            d = subprocess.run([AVIFDEC, "-c", "dav1d", p, png], capture_output=True)
            if d.returncode == 0:
                o = subprocess.run([SSIM2, src, png], capture_output=True, text=True)
                try:
                    r["ssim2"] = float(o.stdout.strip().split()[-1])
                except (ValueError, IndexError):
                    r["ssim2"] = None
                r["psnr"] = psnr_rgb(src, png)
            else:
                r["ssim2"] = r["psnr"] = None
            ivf = os.path.join(td, "a.ivf")
            try:
                avif_to_ivf(p, ivf)
                o = subprocess.run([LRSTAT, ivf], capture_output=True, text=True)
                r["lr"] = json.loads(o.stdout.strip().splitlines()[-1])
            except Exception as e:  # noqa: BLE001
                r["lr"] = None
                r["lr_err"] = str(e)[:200]
        return r

    os.makedirs(f"{LR}/tmp", exist_ok=True)
    t0 = time.time()
    with open(sp, "a") as f, cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            f.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"score done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(sys.argv[2:])
    elif sys.argv[1] == "score":
        score()
