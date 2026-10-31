#!/usr/bin/env python3
"""Stage 4: CPU timings with bench.py's own speed method (bench.cpu_speed: the gamut set repeated to >= 1000 MP,
one timed pass, q60, -y 420, libaom tune=iq / SVT-AV1 tune=3), sha256 of every output.

  timing.py main    stock -j1 reference/control, A (thread splits, stock), B (tuned, PGO, best splits), control
  timing.py r2      second round: stock / tuned / PGO back to back per encoder, and stock 8 x -j2
  timing.py tiles   optional: --autotiling at 1 x -j16 and 4 x -j4 (stock; avifenc's default anyway, it turned out)
  timing.py notiles one tile per image (--tilerowslog2 0 --tilecolslog2 0) vs the default, back to back

Resumable: runs already in ~/out/timing.json are skipped.  Encoded files are kept in ~/keep/<group>_<label>/."""
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import types

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/av1gpu/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

IMG = f"{H}/av1gpu/bench/images_gamut"
WORK, KEEP, OUT = f"{H}/work", f"{H}/keep", f"{H}/out"
BUILDS = {"stock": f"{H}/opt/bin/avifenc", "tuned": f"{H}/opt-tuned/bin/avifenc", "pgo": f"{H}/opt-pgo/bin/avifenc"}
ENC = {"aom_s6": ("aom", 6), "aom_s9": ("aom", 9), "svt_p2": ("svt", 2), "svt_p4": ("svt", 4), "svt_p6": ("svt", 6),
       "svt_p8": ("svt", 8), "svt_p9": ("svt", 9)}
ALL = list(ENC)
A_ENC = ["aom_s6", "aom_s9", "svt_p4", "svt_p6", "svt_p9"]
SPLITS = [1, 2, 4, 8, 16]
TJ = os.path.join(OUT, "timing.json")
HJ = os.path.join(OUT, "hashes.json")


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def label(build, enc, threads, tiles, notiles=False):
    return f"{build}_{enc}_j{threads}" + ("_tiles" if tiles else "") + ("_notiles" if notiles else "")


def main():
    phase = sys.argv[1]
    os.makedirs(KEEP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IMG, "*.jpg")))
    sizes = {os.path.basename(f): (lambda im: im.width * im.height / 1e6)(Image.open(f)) for f in files}
    sdir, sfiles, smp, reps = bench.speed_set(files, sizes, 1000, WORK)
    res = json.load(open(TJ)) if os.path.exists(TJ) else dict(runs=[])
    hashes = json.load(open(HJ)) if os.path.exists(HJ) else {}
    res.update(images=len(files), megapixels=sum(sizes.values()),
               speed_workload=dict(images=len(sfiles), megapixels=smp, repeats=reps))
    print(f"{len(files)} images, {sum(sizes.values()):.1f} MP; speed workload {len(sfiles)} images, {smp:.0f} MP",
          flush=True)
    done = {(r["group"], r["label"]) for r in res["runs"]}

    def ref_digest(enc):
        r = [x for x in res["runs"] if x["build"] == "stock" and x["encoder"] == enc and x["threads"] == 1
             and not x["autotiling"] and not x.get("no_tiling")]
        return r[0]["set_digest"] if r else None

    def run(group, build, enc, threads=1, tiles=False, notiles=False):
        lab = label(build, enc, threads, tiles, notiles)
        if (group, lab) in done:
            return next(x for x in res["runs"] if x["group"] == group and x["label"] == lab)
        codec, speed = ENC[enc]
        a = types.SimpleNamespace(avifenc=BUILDS[build], svt_tune="3", avifenc_threads=threads, autotiling=tiles,
                                  no_tiling=notiles)
        t_start = time.time()
        r = bench.cpu_speed(a, sfiles, speed, True, smp, WORK, codec)
        out = os.path.join(WORK, f"speed_{codec}_{speed}")
        h = {n: sha256(os.path.join(out, n)) for n in sorted(os.listdir(out))}
        per = {}
        for n, v in h.items():
            per.setdefault(n.split("_", 1)[1], set()).add(v)
        r0 = {n.split("_", 1)[1]: v for n, v in h.items() if n.startswith("r0_")}
        r0_bytes = {n.split("_", 1)[1]: os.path.getsize(os.path.join(out, n)) for n in h if n.startswith("r0_")}
        digest = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(r0.items())).encode()).hexdigest()
        dest = os.path.join(KEEP, f"{group}_{lab}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(out, dest)
        ref = ref_digest(enc)
        e = dict(r, group=group, label=lab, build=build, encoder=enc, threads=threads,
                 parallel=bench.avifenc_parallel(a, os.cpu_count()), autotiling=tiles, no_tiling=notiles, started=t_start,
                 deterministic_across_repeats=all(len(s) == 1 for s in per.values()), set_digest=digest,
                 r0_bytes=sum(r0_bytes.values()), identical_to_stock_j1=(digest == ref) if ref else None)
        res["runs"].append(e)
        done.add((group, lab))
        hashes[f"{group}/{lab}"] = dict(sha256=r0, bytes=r0_bytes)
        json.dump(res, open(TJ + ".tmp", "w"), indent=1)
        os.replace(TJ + ".tmp", TJ)
        json.dump(hashes, open(HJ + ".tmp", "w"))
        os.replace(HJ + ".tmp", HJ)
        print(f"{time.strftime('%H:%M:%S')} {group:8s} {lab:24s} {e['mp_s']:7.2f} MP/s  {e['wall_s']:6.1f} s  "
              f"{e['bytes']:>10d} B  ok={e['images']}  det={e['deterministic_across_repeats']}  "
              f"same_as_stock_j1={e['identical_to_stock_j1']}", flush=True)
        return e

    if phase == "main":
        for enc in ALL:                       # stock -j1: control (vs the published speeds) and byte reference
            run("stock_j1", "stock", enc)
        for enc in A_ENC:                     # A: thread splits, stock
            for t in SPLITS:
                run("A", "stock", enc, t)
        for enc in ALL:                       # B: tuned
            run("B", "tuned", enc)
        for enc in ALL[2:]:                   # B: tuned + PGO (SVT-AV1 only; libaom is opt-tuned's)
            run("B", "pgo", enc)
        for enc in A_ENC:                     # B: tuned at the best split from A, if that is not -j1
            a_runs = [x for x in res["runs"] if x["group"] == "A" and x["encoder"] == enc]
            best = max(a_runs, key=lambda x: x["mp_s"])
            if best["threads"] != 1:
                run("B_split", "stock", enc, best["threads"])
                run("B_split", "tuned", enc, best["threads"])
                if enc.startswith("svt"):
                    run("B_split", "pgo", enc, best["threads"])
        for enc in ("aom_s6", "svt_p6"):      # control at the end of the session
            run("control_end", "stock", enc)
    elif phase == "r2":                       # second round, builds back to back per encoder (drift cancels)
        for enc in ("aom_s6", "aom_s9", "svt_p4", "svt_p6", "svt_p8", "svt_p9"):
            for b in ("stock", "tuned") + (("pgo",) if enc.startswith("svt") else ()):
                run("R2", b, enc)
        for enc in ("aom_s6", "svt_p4"):
            run("R2", "stock", enc, 2)
    elif phase == "notiles":                  # avifenc tiles by default: the same encoders with one tile per image
        for enc in ("aom_s6", "aom_s9", "svt_p4", "svt_p6", "svt_p9"):
            run("NT", "stock", enc)
            run("NT", "stock", enc, 1, notiles=True)
        for enc in ("aom_s6", "svt_p4"):
            run("NT", "stock", enc, 2, notiles=True)
    elif phase == "tiles":
        for enc in A_ENC:
            for t in (16, 4):
                run("tiles", "stock", enc, t, True)
    print(f"PHASE_DONE {phase}", flush=True)


if __name__ == "__main__":
    main()
