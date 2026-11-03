#!/usr/bin/env python3
"""CVVDP (cvvdp 0.5.7, torch CPU, display standard_fhd) of the 4:2:0 files of s6, p6keep, p6full, s4 and s4r1, through
bench.py's Cvvdp class exactly as bench.py --cvvdp scores (8-bit sRGB via LittleCMS, 1:1 pixels), on the dav1d decode.
One process per worker (CV_WORKERS, default one per vCPU), CV_THREADS torch threads each (default 1); about
0.5 GB per MP of source per process (9 GB at 17.9 MP), so the large photos run with fewer, wider workers.
Run with the venv's python (~/cv/bin/python).

  cvscore.py MIN_MP MAX_MP [CFG ...]    sources with MIN_MP <= MP < MAX_MP   -> appended to ~/lr/rows/cvvdp.jsonl"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

H = os.path.expanduser("~")
LR = f"{H}/lr"
SETS = {"std": f"{H}/img/std", "gamut": f"{H}/img/gamut", "kodak": f"{H}/img/kodak"}
CFGS = ["s6", "p6keep", "p6full", "s4", "s4r1"]
QS = [30, 45, 60, 75, 90]
_M = []


def init():
    import torch
    torch.set_num_threads(int(os.environ.get("CV_THREADS", "1")))
    sys.path.insert(0, f"{H}/bench")
    import bench
    _M.append(bench.Cvvdp("standard_fhd", "cpu"))


def one(j):
    c, s, src, q, p = j
    r = dict(fmt="420", cfg=c, set=s, image=os.path.basename(src), q=q, bytes=os.path.getsize(p), jod=None)
    with tempfile.TemporaryDirectory(dir=f"{LR}/tmp") as td:
        png = os.path.join(td, "d.png")
        if subprocess.run([f"{H}/opt/bin/avifdec", "-c", "dav1d", p, png], capture_output=True).returncode == 0:
            r["jod"] = _M[0](src, png, p)
    return r


def mp(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size[0] * im.size[1] / 1e6


if __name__ == "__main__":
    min_mp, max_mp = float(sys.argv[1]), float(sys.argv[2])
    cfgs = sys.argv[3:] or CFGS
    workers = int(os.environ.get("CV_WORKERS", os.cpu_count()))
    sizes = {}
    jobs = []
    done = set()   # rows already in cvvdp.jsonl (a restarted band skips them)
    if os.path.exists(f"{LR}/rows/cvvdp.jsonl"):
        for l in open(f"{LR}/rows/cvvdp.jsonl"):
            r = json.loads(l)
            done.add((r["cfg"], r["set"], r["image"], r["q"]))
    for c in cfgs:
        for s, d in SETS.items():
            for f in sorted(os.listdir(d)):
                src = os.path.join(d, f)
                if src not in sizes:
                    sizes[src] = mp(src)
                if not min_mp <= sizes[src] < max_mp:
                    continue
                for q in QS:
                    if (c, s, f, q) not in done:
                        jobs.append((c, s, src, q, f"{LR}/out/420/{c}/{s}/{f}.q{q}.avif"))
    jobs.sort(key=lambda j: (cfgs.index(j[0]), -sizes[j[2]]))   # configurations in the order given
    print(len(jobs), "jobs", flush=True)
    with open(f"{LR}/rows/cvvdp.jsonl", "a") as f, cf.ProcessPoolExecutor(workers, initializer=init) as ex:
        for r in ex.map(one, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("CVVDP_DONE", flush=True)
